/** @file  spectralDrawing.cpp
 * 
 *  @brief Given a graph as .csr, generate .nxyz file with x,y coordinates
 *
 */
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <assert.h>
#include <chrono>
#include <queue>
#include <set>
#include <getopt.h>
#include <string.h>
#include <omp.h>
#include <stdlib.h>
#include "PivotMDS.h"
#include "PHDE.h"
#include "WParHDE.h"
#include "ParHDE.h"

typedef struct
{
  long n;
  long m;
  unsigned int *rowOffsets;
  unsigned int *adj;
} graph_t;

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 8)
{

  std::ostringstream out;
  out << std::setprecision(n) << a_value;
  return out.str();
}

static int
writeCoords(int n,
            double *coordinates,
            int numCoord,
            int coarseningType, int doHDE, int refineType,
            double eps, char *inputFilename)
{

  // Write coordinates to file
  std::ofstream fout;
  std::string coordFilename(inputFilename);
  coordFilename +=  "_hde.nxyz";
  std::cout << "Writing coordinates to file " << coordFilename << std::endl;
  fout.open(coordFilename);
  for (int i = 0; i < n; i++)
  {
    for(int j=0; j < numCoord-1; j++)
    {
      fout<<coordinates[i+j*n]<<",";
    }
    fout<<coordinates[i+(numCoord-1)*n]<<std::endl;
  }
  fout.close();
  return 0;
}

int main(int argc, char **argv)
{
  char *inputFilename = argv[1];
  int weighted = 0;
  int numCoord = 2;
  int doRand = 2;
  int drawingAlgorithm = 3;
  int option_index = 0;
  int numOpts = 0;
  int numSources = 10;
  while (( option_index = getopt(argc, argv, "w:hc:a:p:r:")) != -1)
  {
    numOpts++;
    switch (option_index) 
    {
      case 'w':
        weighted = 1;
        break;
      case 'c':
        numCoord = atoi(optarg);
        break;
      case 'p':
        if(!strcmp(optarg, "RandFine"))
          doRand = 1;
        else if(!strcmp(optarg, "RandCoarse"))
          doRand = 0;
        else if(!strcmp(optarg, "Kcenters"))
          doRand = 2;
        else
        {
          std::cerr << "Unrecognized Strategy" <<  std::endl;
          abort();
        }
        break;
      case 'a':
        if(!strcmp(optarg, "PivotMDS"))
          drawingAlgorithm = 0;
        else if(!strcmp(optarg, "PHDE"))
          drawingAlgorithm = 1;
        else if(!strcmp(optarg, "wParHDE"))
          drawingAlgorithm = 2;
        else if(!strcmp(optarg, "ParHDE"))
          drawingAlgorithm = 3;
        else
        {
          std::cerr << "Unrecognized algorithm" <<  std::endl;
          abort();
        }
        break;
      case 'r':
        numSources = atoi(optarg);
        break;
      case 'h':
        std::cout << "Usage: " << argv[0] << " [csr filename]\n"
                                         "-w         : Indicates if CSR file is weighted or unweighted (default)\n"
                                         "-c <value> : number of coordinates to generate( default=2)\n"
                                         "-a <algo>  : Indicate the algorithm to run (PivotMDS,PHDE, wParHDE, ParHDE(default))\n"
                                         "-p <strat> : Pivot picking strategy to use (RandFine,RandCoarse,Kcenters(default))\n"
                                         "-r <value> : Number of pivots (default=10)"
                                         "-h         : help"
              << std::endl;
        exit(1);
      default:
        abort();
      return 1;
     } //end block for switch
   }  //end block for while

  // Read CSR file
  auto startTimer = std::chrono::high_resolution_clock::now();
  auto startTimerPart = std::chrono::high_resolution_clock::now();
  FILE *infp = fopen(inputFilename, "rb");
  if (infp == NULL)
  {
    std::cout << "Error: Could not open input file. Exiting ..." << std::endl;
    return 1;
  }
  long n, m;
  long rest[4];
  unsigned int *rowOffsets, *adj;
  int32_t *weights;
  fread(&n, 1, sizeof(long), infp);
  fread(&m, 1, sizeof(long), infp);
  fread(rest, 4, sizeof(long), infp);
  rowOffsets = (unsigned int *)malloc(sizeof(unsigned int) * (n + 1));
  assert(rowOffsets != NULL);
  adj = (unsigned int *)malloc(sizeof(unsigned int) * m);
  assert(adj != NULL);
  fread(rowOffsets, n + 1, sizeof(unsigned int), infp);
  fread(adj, m, sizeof(unsigned int), infp);
  if(weighted)
  {
    weights = (int *)malloc(sizeof(int32_t) * m);
    fread(weights, m, sizeof(int32_t), infp);
  }
  else
    weights = NULL;
  fclose(infp);
  auto endTimerPart = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elt = endTimerPart - startTimerPart;
  std::cout << "CSR read time: " << elt.count() << " s." << std::endl;
  std::cout << "Num edges: " << m / 2 << ", vertices: " << n << ", nnz: "<< m << std::endl;

  graph_t g;
  g.n = n;
  g.m = m;
  g.rowOffsets = rowOffsets;
  g.adj = adj;

  DrawingBase *draw;
  switch(drawingAlgorithm)
  {
    case 0: draw = new PivotMDS(g.rowOffsets, g.adj, NULL, g.n, g.m, numSources, doRand, numCoord);
            break;
    case 1: draw = new PHDE(g.rowOffsets, g.adj, NULL, g.n, g.m, numSources, doRand,numCoord);
            break;
    case 2: draw = new WParHDE(g.rowOffsets, g.adj, weights, g.n, g.m, numSources, numCoord);
            break;
    case 3: draw = new ParHDE(g.rowOffsets, g.adj, NULL, g.n, g.m, numSources, doRand, numCoord);
            break;
    default: std::cerr << "Invalid argument: Select correct algorithm" <<std::endl;
             std::exit(0);
  }

  draw->run(1);
  double *coordinates = draw->getCoordinates();
  writeCoords(g.n, coordinates, numCoord, 0, 1, 0, 0, inputFilename);
  free(g.rowOffsets);
  free(g.adj);
  if(weighted)
    free(weights);
  endTimerPart = std::chrono::high_resolution_clock::now();
  elt = endTimerPart - startTimer;
  std::cout << "Overall time: " << elt.count() << " s." << std::endl;

  return 0;
}