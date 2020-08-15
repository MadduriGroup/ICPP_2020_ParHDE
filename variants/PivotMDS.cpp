/** @file  PivotMDS.cpp
 * 
 *  @brief Parallel implementation of PivotMDS
 *
*/
#include "PivotMDS.h"
#include <cmath>
#include <chrono>
#include <iostream>

PivotMDS::PivotMDS(unsigned int *rowPtrs, unsigned int *adj, int32_t *weights, long n, long m, int numSources, int doRand, int numCoord)
: DrawingBase(rowPtrs, adj, weights, n, m, numSources, doRand, numCoord)
{
  #ifndef INTEL_MKL
  this->matrixC = (double *)malloc(sizeof(double) * n * numSources);
  #else
  this->matrixC = (double *)mkl_malloc(sizeof(double) * n * numSources, ALIGN);
  #endif
}

PivotMDS::~PivotMDS()
{
  #ifndef INTEL_MKL
  free(matrixC);
  #else
  mkl_free(matrixC);
  #endif
}

void PivotMDS::doubleCentering(double *delta)
{
  // Calculating sums of squares
  double *deltaColumnSum = new double[numSources];
  double *deltaRowSum = (double *)malloc(sizeof(double) * n);
  double deltaTotalSum = 0;
  #pragma omp parallel for schedule(guided)
  for(int i=0; i < n; i++)
    deltaRowSum[i] = 0;
  for(int i=0; i < numSources; i++)
  {
    double columnSum = 0 ;
    #pragma omp parallel for reduction(+:columnSum) schedule(static)
    for(int j=0; j < n; j++)
    {
      columnSum += pow(delta[j + i*n], 2);
      deltaRowSum[j] += pow(delta[j + i*n], 2);
    }
    deltaColumnSum[i] = columnSum;
    deltaTotalSum += columnSum;
  }
  double reciprocalN = (double)1 / (double)n;
  double reciprocalK = (double)1 / (double)numSources;
  #pragma omp parallel for collapse(2) schedule(guided)
  for(int i=0; i < numSources; i++)
  {
    for(int j=0; j < n; j++)
    {
      this->matrixC[j + i*n] = -0.5 * (pow(delta[j + i*n],2) -
                                      (reciprocalN * deltaColumnSum[i]) -
                                      (reciprocalK * deltaRowSum[j]) +
                                      (reciprocalK * reciprocalN * deltaTotalSum));
      }
  }
  free(deltaRowSum);
  delete [] deltaColumnSum;
}

int PivotMDS::run(int tries = 1, int alpha = 15, int beta = 18, int32_t delta = 1)
{
  std::cout<<"Performing PivotMDS with "<<tries<<" tries and pivots: "<<numSources<<std::endl;
  #pragma omp parallel
  {
    #pragma omp single
    printf("Number of threads launched : %d \n", omp_get_num_threads());
  }
  double *eltDistance = new double[tries], *eltColumnCentering = new double[tries],
         *eltMMult = new double[tries], *eltOthers = new double[tries];
  auto startInnerTimer = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elt;
  #ifndef INTEL_MKL
  double *Distances = (double *) malloc (sizeof(double) * n * numSources);
  double *cTransC = (double *) malloc (sizeof(double) * numSources * numSources);
  #else
  double *Distances = (double *) mkl_malloc (sizeof(double) * n * numSources, ALIGN);
  double *cTransC = (double *) mkl_malloc (sizeof(double) * numSources * numSources, ALIGN);
  #endif
  for(int i=0 ; i < tries; i++)
  {
    eltDistance[i] = eltColumnCentering[i] = eltMMult[i] = eltOthers[i] = 0;
  }
  for(int trial=0; trial < tries; trial++)
  {
    // k-centered distances
    distancePhaseComputation(&Distances, eltDistance, trial, alpha, beta, delta);
          
    // double centering
    startInnerTimer = std::chrono::high_resolution_clock::now();
    doubleCentering(Distances);
    elt = std::chrono::high_resolution_clock::now() - startInnerTimer;
    eltColumnCentering[trial] += elt.count();

    // C'C matrix multiplication
    startInnerTimer = std::chrono::high_resolution_clock::now();
    cTransCmatrixComputation(cTransC);
    elt = std::chrono::high_resolution_clock::now() - startInnerTimer;
    eltMMult[trial] += elt.count();

    startInnerTimer = std::chrono::high_resolution_clock::now();
    compute_eigens(cTransC, numSources, this->eigenVectors, Largest);
    this->computeCoordinates(this->matrixC);
    elt = std::chrono::high_resolution_clock::now() - startInnerTimer;
    eltOthers[trial] += elt.count();
  }

  #ifndef INTEL_MKL
  free(Distances);
  free(cTransC);
  #else
  mkl_free(Distances);
  mkl_free(cTransC);
  #endif
  double totalBfs = 0, totalHDE = 0, totalMMult = 0,
         meanBFS, meanTotal, meanMMult, totalCCenter = 0, meanCCenter;
  for (int t = 0; t < tries; t++)
  {
    totalBfs += eltDistance[t];
    totalHDE += eltColumnCentering[t] + eltDistance[t] + eltOthers[t] + eltMMult[t];
    totalMMult += eltMMult[t];
    totalCCenter += eltColumnCentering[t];
  }
  meanBFS = totalBfs / tries;
  meanTotal = totalHDE / tries;
  meanMMult = totalMMult / tries;
  meanCCenter = totalCCenter / tries;
  totalBfs = totalHDE = totalMMult = totalCCenter = 0;

  std::cout << "**** PivotMDS Means ******" << std::endl;
  std::cout << "Total mean time " << meanTotal << " s." << std::endl;
  std::cout << "BFS mean time " << meanBFS << " s." << std::endl;
  std::cout << "Matrix Multiply Mean time " << meanMMult << " s." << std::endl;
  std::cout << "Double Centering Mean time " << meanCCenter << " s." << std::endl;
  delete [] eltDistance;
  delete [] eltColumnCentering;
  delete [] eltMMult;
  delete [] eltOthers;
  return 0;
}

void PivotMDS::cTransCmatrixComputation(double *cTransC)
{
  matrixTransMultiplication(this->matrixC, this->matrixC, cTransC, numSources, n, numSources);
}

