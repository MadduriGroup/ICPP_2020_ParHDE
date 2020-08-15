/** @file  ParHDE.cpp
 * 
 *  @brief Parallel implementation of HDE.
 *
*/
#include "ParHDE.h"
#include <cmath>
#include <chrono>
#include <iostream>
#include <set>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define S_MAX 51

ParHDE::ParHDE(unsigned int *rowPtrs, unsigned int *adj, int32_t *weights, long n, long m, int numSources, int doRand, int numCoord)
: DrawingBase(rowPtrs, adj, weights, n, m, numSources, doRand, numCoord)
{
  #ifndef INTEL_MKL
  this->LSmatrix = (double *)malloc(sizeof(double) * n * numSources);
  this->SLSmatrix = (double *)malloc(sizeof(double) * numSources * numSources);  
  #else
  this->LSmatrix = (double *)mkl_malloc(sizeof(double) * n * numSources, ALIGN);
  this->SLSmatrix = (double *)mkl_malloc(sizeof(double) * numSources * numSources, ALIGN);
  #endif
  if(weights)
  {
    std::cerr<<"ParHDE code doesn't requires the weighted edges"<<std::endl;
    exit(0);
  }
}

ParHDE::~ParHDE()
{
  #ifndef INTEL_MKL
  free(SLSmatrix);
  free(LSmatrix);
  #else
  mkl_free(SLSmatrix);
  mkl_free(LSmatrix);
  #endif
}

int ParHDE::run(int tries = 1, int alpha = 15, int beta = 18, int32_t delta = 1)
{
  #pragma omp parallel
  {
    #pragma omp single
    printf("Running ParHDE\nNumber of threads launched : %d \n", omp_get_num_threads());
  }

  #ifndef INTEL_MKL
  double *degrees = (double *)malloc(this->n * sizeof(double));
  #else
  double *degrees = (double *)mkl_malloc(this->n * sizeof(double), ALIGN);
  #endif
  for (long i = 0; i < n; i++)
    degrees[i] = (double)(this->rowPtrs[i + 1] - this->rowPtrs[i]);

  double *eltHDE= new double[tries], *eltDistance = new double[tries],
         *eltOrth = new double[tries], *eltMMult = new double[tries];
  for (int t = 0; t < tries; t++)
    eltHDE[t] = eltDistance[t] = eltOrth[t] = eltMMult[t] = 0;

  auto startTimerPart = std::chrono::high_resolution_clock::now();
  auto startTimerPartTemp = std::chrono::high_resolution_clock::now();
  auto endTimerPart = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elt = endTimerPart - startTimerPart;
  /* Sources to compute the BFSs from */
  int pivotsLimits = 3 * numSources;
  std::cout << "Performing ParHDE with Number of sources: " << numSources << " and ";
  std::cout << "Number of experiments: " << tries << std::endl;

  #ifndef INTEL_MKL
  double *distMatrix = (double *)malloc(n * numSources * sizeof(double));
  // double *distUnfilteredMatrix = (double *)malloc(n * (numSources + 1) * sizeof(double));
  double *distUnfilteredMatrix = (double *)malloc(n * (pivotsLimits + 1) * sizeof(double));
  #else
  double *distMatrix = (double *)mkl_malloc(n * numSources * sizeof(double), ALIGN);
  double *distUnfilteredMatrix = (double *)mkl_malloc(n * (pivotsLimits + 1) * sizeof(double), ALIGN);
  #endif
  double *min_dists = (double *)malloc(n * sizeof(double));
  double mult_denom[8*S_MAX];
  double mult_num[8*S_MAX];
  double mult_denom_inv[8*S_MAX];

  /* 10 Tries of the HDE algorithm */
  for (int t = 0; t < tries; t++)
  {
    // HDE Initialize
    startTimerPart = std::chrono::high_resolution_clock::now();
    int maxDegree = 0;
    for (long p = 0; p < n; p++)
    {
      distUnfilteredMatrix[p] = 1;
      min_dists[p] = INT_MAX;
      if(maxDegree < degrees[p])
    	  maxDegree = degrees[p];
    }
    double
    #ifndef INTEL_MKL
    norm = 1 / parComp.normL2(n, distUnfilteredMatrix);
    #else
    norm = 1 / cblas_dnrm2(n, distUnfilteredMatrix, 1);
    #endif
    for (long p = 0; p < n; p++)
      distUnfilteredMatrix[p] = distUnfilteredMatrix[p] * norm;
    startTimerPartTemp = std::chrono::high_resolution_clock::now();
    auto startInnerTimer = std::chrono::high_resolution_clock::now();
    long start_idx;
    start_idx = 0;
    srand(time(NULL));
    
    std::set<long> sources; // container to store unique start indices
    
    // BFS De-copupled phases
    if(doRand==2)
    {
      for (int run_count = 1; run_count <= pivotsLimits; run_count++)
      {
        long old_idx;
        this->DistanceAlgorithm(rowPtrs, adj, start_idx, n, m, (distUnfilteredMatrix + run_count * n));
        struct Compare max_dist;
        max_dist.val = -1;
        max_dist.index = 0;
        #pragma omp parallel shared(max_dist)
        {
          #pragma omp for reduction(argmax : max_dist)
          for (long i = 0; i < n; i++)
          {
            if (distUnfilteredMatrix[i + run_count * n] < min_dists[i])
              min_dists[i] = distUnfilteredMatrix[i + run_count * n];
            if (min_dists[i] > max_dist.val)
            {
              max_dist.val = min_dists[i];
              max_dist.index = i;
            }
          }
        }

        start_idx = max_dist.index;
        old_idx = start_idx;
        if (run_count != 1)
        {
          if (sources.find(start_idx) != sources.end())
          { // source is not unique
            #pragma omp parallel for
            for (long i = 0; i < n; i++)
              if (min_dists[i] == max_dist.val)       // find next source
                if (sources.find(i) == sources.end()) // source is unique
                  start_idx = i;
           }
        }
        sources.insert(old_idx);
      }
    }
    else
    {
      unsigned *sources = new unsigned[numSources];
      for(int run_count = 0; run_count < pivotsLimits; run_count++)
        sources[run_count] = rand() % n;
      if(doRand==0) // rand pivots for small graphs
      {
        #pragma omp parallel for
        for (int run_count = 1; run_count <= pivotsLimits; run_count++)
          this->seqBfs(sources[run_count-1], (distUnfilteredMatrix + run_count * n));
      }
      else  // rand pivots for large graphs
      {
        for (int run_count = 1; run_count <= pivotsLimits; run_count++)
          this->DistanceAlgorithm(rowPtrs, adj, sources[run_count-1], n, m, (distUnfilteredMatrix + run_count * n));
      }
      delete [] sources;
    }
    #pragma omp parallel for
    for (int run_count = 1; run_count <= pivotsLimits; run_count++)
    {
      double normdist = 0;
      for (long p = 0; p < n; p++)
        normdist += pow(distUnfilteredMatrix[p + run_count * n], 2);
      normdist = 1 / sqrt(normdist);
      for (long p = 0; p < n; p++)
        distUnfilteredMatrix[p + run_count * n] = distUnfilteredMatrix[p + run_count * n] * normdist;
    }
    endTimerPart = std::chrono::high_resolution_clock::now();
    elt = endTimerPart - startTimerPart;
    eltDistance[t] += elt.count();
    
    // BLAS CGS Decoupled 
    startInnerTimer = std::chrono::high_resolution_clock::now();
    int correct_pivots = 1;
    for (int run_count = 1; run_count <= pivotsLimits; run_count++)
    {
      if(correct_pivots > numSources) break;
      int j = correct_pivots;
      // Orthogonalizing with vector rest of the settled vectors
      for (int k=0; k < S_MAX; k++)
      {
        mult_denom[8*k]     = 0.0;
        mult_num[8*k]       = 0.0;
        mult_denom_inv[8*k] = 0.0;
      }
      #pragma omp parallel
      {
        double mult_denom_l[S_MAX];
        double mult_num_l[S_MAX];
        for (unsigned int k=0; k < S_MAX; k++) 
        {
          mult_denom_l[k]   = 0.0;
          mult_num_l[k]     = 0.0;
        }

        const double * __restrict__ Ac = distMatrix;
        const double *__restrict__ An = distUnfilteredMatrix;
        #pragma omp for schedule(static)
        for(int ii=0 ; ii < n; ii++) 
        {
          double An_ii_j = An[ii+n*run_count];
          for(int k=0; k < j-1 ; k++) 
          {
            mult_num_l[k]   += An_ii_j * Ac[ii + n*k] * degrees[ii];
            mult_denom_l[k] += Ac[ii + n*k] * Ac[ii + n*k] * degrees[ii];
          }
          mult_num_l[S_MAX-1]   += An_ii_j * An[ii]* degrees[ii];
          mult_denom_l[S_MAX-1] += An[ii] * An[ii] * degrees[ii];
        }
        #pragma omp critical
        {
          for (int k=0; k < j-1; k++) 
          {
            mult_denom[8*k]     += mult_denom_l[k];
            mult_num[8*k]       += mult_num_l[k];
          }
          mult_denom[8*(S_MAX-1)] += mult_denom_l[S_MAX-1];
          mult_num[8*(S_MAX-1)]   += mult_num_l[S_MAX-1];
        }
      }
      for (int k=0; k < j-1; k++) 
      {
        mult_denom_inv[8*k] = 1/mult_denom[8*k];
      }

      mult_denom_inv[8*(S_MAX-1)] = 1 / mult_denom[8*(S_MAX-1)];
      #pragma omp parallel 
      {
        const double * __restrict__ Ac = distMatrix;
        const double *__restrict__ An = distUnfilteredMatrix;
        #pragma omp for schedule(static)
        for(unsigned ii=0; ii < n; ii++)
        {
          double accum_sum=0;
          for(int k=0; k < j-1; k++)
          {
            accum_sum += mult_num[8*k] * mult_denom_inv[8*k] * Ac[ii + n*k];
          }
          accum_sum += mult_num[8*(S_MAX-1)] * mult_denom_inv[8*(S_MAX-1)] * An[ii];
          distUnfilteredMatrix[ii + n*run_count] -= accum_sum;
        }
      }
      double 
      #ifdef INTEL_MKL
      normdist = cblas_dnrm2(n, distUnfilteredMatrix + run_count * n, 1);
      #else
      normdist = parComp.normL2(n, distUnfilteredMatrix + run_count * n);
      #endif
      if (normdist < 0.001)
        std::cout << "discarding vec " << j << ", normdist " << normdist << std::endl;
      else
      {
        normdist = 1 / normdist;
        #pragma omp parallel for simd
        for (long p = 0; p < n; p++)
          distMatrix[p + (j - 1) * n] = distUnfilteredMatrix[p + run_count * n] * normdist;
        correct_pivots++;
      }
    } 
    endTimerPart = std::chrono::high_resolution_clock::now();
    elt = endTimerPart - startInnerTimer;
    eltOrth[t] += elt.count();
    if(correct_pivots < numSources)
    {
      std::cerr<<"Number of Pivots obtained: "<<correct_pivots<<" inadequate, increase the Pivots Limits: "<<pivotsLimits<<std::endl;
      exit(0);
    } 
    startTimerPartTemp = std::chrono::high_resolution_clock::now();
    /* Multiplication of Laplacians with untransposed distance matrix S */
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int k = 0; k < numSources; k++)
    {
     for (long i = 0; i < n; i++)
     {
        this->LSmatrix[i + k * n] = degrees[i] * distMatrix[i + k * n];
        for (unsigned int j = rowPtrs[i]; j < rowPtrs[i + 1]; j++)
        {
          unsigned int v = this->adj[j];
          this->LSmatrix[i + k * n] -= distMatrix[v + k * n];
        }
      }
    }
    matrixTransMultiplication(distMatrix, this->LSmatrix, this->SLSmatrix, numSources, n, numSources);
    endTimerPart = std::chrono::high_resolution_clock::now();
    elt = endTimerPart - startTimerPartTemp;
    eltMMult[t] = elt.count();
    /* Eigen value-vector computation */
    compute_eigens(this->SLSmatrix, numSources, this->eigenVectors, First);
    /* Multipying distance matrix with first two eigen vectors */
    this->computeCoordinates(distMatrix);
    endTimerPart = std::chrono::high_resolution_clock::now();
    elt = endTimerPart - startTimerPart;
    eltHDE[t] = elt.count();
  } // end of tries
  free(min_dists);
  #ifndef INTEL_MKL
  free(distUnfilteredMatrix);
  free(distMatrix);
  #else
  mkl_free(distUnfilteredMatrix);
  mkl_free(distMatrix);
  #endif
  double totalBfs = 0, totalHDE = 0, totalOrth = 0,
         totalMMult = 0, meanBFS, meanHDE, meanOrth, meanMMult;
  for (int t = 0; t < tries; t++)
  {
    totalBfs += eltDistance[t];
    totalHDE += eltHDE[t];
    totalOrth += eltOrth[t];
    totalMMult += eltMMult[t];
  }
  meanBFS = totalBfs / tries;
  meanHDE = totalHDE / tries;
  meanOrth = totalOrth / tries;
  meanMMult = totalMMult / tries;
  totalBfs = totalHDE = totalOrth = totalMMult = 0;

  std::cout << "**** Means ******" << std::endl;
  std::cout << "HDE mean time " << meanHDE << " s." << std::endl;
  std::cout << "BFS mean time " << meanBFS << " s." << std::endl;
  std::cout << "DOrthogonalize mean time " << meanOrth << " s." << std::endl;
  std::cout << "Matrix Multiply Mean time " << meanMMult << " s." << std::endl;

  delete [] eltHDE;
  delete [] eltDistance;
  delete [] eltOrth;
  delete [] eltMMult;
  #ifndef INTEL_MKL
  free(degrees);
  #else
  mkl_free(degrees);
  #endif
  return 0;
}
