/** @file  ParHDE.cpp
 * 
 *  @brief Parallel implementation of HDE for weighted graphs.
 *
*/
#include "WParHDE.h"
#include <cmath>
#include <chrono>
#include <iostream>
#include <set>

WParHDE::WParHDE(unsigned int *rowPtrs, unsigned int *adj, int32_t *weights, long n, long m, int numSources, int numCoord)
: DrawingBase(rowPtrs, adj, weights, n, m, numSources, numCoord)
{
   #ifndef INTEL_MKL
  this->LSmatrix = (double *)malloc(sizeof(double) * n * numSources);
  this->SLSmatrix = (double *)malloc(sizeof(double) * numSources * numSources);
  #else
  this->LSmatrix = (double *)mkl_malloc(sizeof(double) * n * numSources, ALIGN);
  this->SLSmatrix = (double *)mkl_malloc(sizeof(double) * numSources * numSources, ALIGN);
  #endif
  if(!weights)
  {
    std::cerr<<"Weighted ParHDE code requires the weighted edges"<<std::endl;
    exit(0);
  }
}

WParHDE::~WParHDE()
{
  #ifndef INTEL_MKL
  free(SLSmatrix);
  free(LSmatrix);
  #else
  mkl_free(SLSmatrix);
  mkl_free(LSmatrix);
  #endif
}

int WParHDE::run(int tries = 11, int alpha = 15, int beta = 18, int32_t delta = 1)
{
  #pragma omp parallel
  {
    #pragma omp single
    printf("Running Weighted ParHDE\nNumber of threads launched : %d \n", omp_get_num_threads());
  }

  #ifdef INTEL_MKL
  double *degrees = (double *)mkl_malloc(this->n * sizeof(double), ALIGN);
  #else
  double *degrees = (double *)malloc(this->n * sizeof(double));
  #endif
  for (unsigned int i = 0; i < n; i++)
  {
    degrees[i] = 0;
    for(unsigned int k = this->rowPtrs[i]; k < this->rowPtrs[i+1]; k++)
      degrees[i] += (double)weights[k];
  }
  double *eltHDE = new double[tries], *eltDistance = new double[tries],
         *eltOrth = new double[tries], *eltMMult = new double[tries];
  for (int t = 0; t < tries; t++)
    eltHDE[t] = eltDistance[t] = eltOrth[t] = eltMMult[t] = 0;

  auto startTimerPart = std::chrono::high_resolution_clock::now();
  auto startTimerPartTemp = std::chrono::high_resolution_clock::now();
  auto endTimerPart = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elt = endTimerPart - startTimerPart;
  /* Sources to compute the BFSs from */
  int pivotsLimits = 3 * numSources;
  std::cout << "Performing parHDE with Number of sources: " << numSources << " and ";
  std::cout << "Number of experiments: " << tries << std::endl;
  #ifndef INTEL_MKL
  double *distMatrix = (double *)malloc(n * numSources * sizeof(double));
  double *distUnfilteredMatrix = (double *)malloc(n * (numSources + 1) * sizeof(double));
  #else
  double *distMatrix = (double *)mkl_malloc(n * numSources * sizeof(double), ALIGN);
  double *distUnfilteredMatrix = (double *)mkl_malloc(n * (numSources + 1) * sizeof(double), ALIGN);
  #endif
  double *min_dists = (double *)malloc(n * sizeof(double));
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
      distUnfilteredMatrix[p] *= norm;
    std::cout<<"maxDegree: "<<maxDegree<<std::endl;

    // Set the delta value for SSSP delta algorithm
    int32_t delta = 10000;
    std::cout<<"Delta: "<<delta<<std::endl;

    /* BFS of maxM sources */
    startTimerPartTemp = std::chrono::high_resolution_clock::now();
    auto startInnerTimer = std::chrono::high_resolution_clock::now();
    long start_idx = 0;
    std::set<long> sources; // container to store unique start indices
    int correct_pivots = 1;
    for (int run_count = 1; run_count <= pivotsLimits; run_count++)
    {
      if(correct_pivots > numSources) break;
      this->DistanceAlgorithm(rowPtrs, adj, start_idx, n, m, (distUnfilteredMatrix + correct_pivots * n), weights, delta);
      struct Compare max_dist;
      max_dist.val = -1;
      max_dist.index = 0;
      #pragma omp parallel shared(max_dist)
      {
        #pragma omp for reduction(argmax : max_dist)
        for (long i = 0; i < n; i++)
        {
          if (distUnfilteredMatrix[i + correct_pivots * n] < min_dists[i])
          {
            min_dists[i] = distUnfilteredMatrix[i + correct_pivots * n];
          }
          if (min_dists[i] > max_dist.val)
          {
            max_dist.val = min_dists[i];
            max_dist.index = i;
          }
        }
      }

      start_idx = max_dist.index;
      long old_idx = start_idx;
      if (correct_pivots != 1)
      {
        if (sources.find(start_idx) != sources.end())
        { // source is not unique
          #pragma omp parallel for
          for (long i = 0; i < n; i++)
          {
            if (min_dists[i] == max_dist.val)       // find next source
              if (sources.find(i) == sources.end()) // source is unique
                start_idx = i;
          }
        }
      }

      sources.insert(old_idx);
      double 
      #ifdef INTEL_MKL
      normdist = 1 / cblas_dnrm2(n, distUnfilteredMatrix + correct_pivots * n, 1);
      #else
      normdist = 1 / parComp.normL2(n, distUnfilteredMatrix + correct_pivots * n);
      #endif
      #pragma omp parallel for simd
      for (long p = 0; p < n; p++)
        distUnfilteredMatrix[p + correct_pivots * n] = distUnfilteredMatrix[p + correct_pivots * n] * normdist;
      endTimerPart = std::chrono::high_resolution_clock::now();
      elt = endTimerPart - startInnerTimer;
      eltDistance[t] += elt.count();
      
      // Coupled D-orthogonalization with MGS version
      startInnerTimer = std::chrono::high_resolution_clock::now();
      int j = correct_pivots;
      for (int k = 0; k < j; k++)
      {
        double multplr_denom = 0, multplr_num = 0;
        #pragma omp parallel
        {
          #pragma omp for simd reduction(+ : multplr_denom, multplr_num)
          for (long p = 0; p < n; p++)
          {
            double dnorm = distUnfilteredMatrix[p + k * n] * degrees[p];
            multplr_denom += distUnfilteredMatrix[p + k * n] * dnorm;
            multplr_num += distUnfilteredMatrix[p + j * n] * dnorm;
          }
          #pragma omp single
          multplr_denom = 1 / multplr_denom;
          #pragma omp for simd
          for (long p = 0; p < n; p++)
          {
            distUnfilteredMatrix[p + j * n] -= (multplr_num * distUnfilteredMatrix[p + k * n] * multplr_denom);
          }
        }
      }

      #ifdef INTEL_MKL
      normdist = cblas_dnrm2(n, distUnfilteredMatrix + j * n, 1);
      #else
      normdist = parComp.normL2(n, distUnfilteredMatrix + j * n);
      #endif
      if (normdist < 0.001)
      {
        std::cout << "discarding vec " << j << ", normdist " << normdist << std::endl;
        j--;
        correct_pivots--;
      }
      else
      {
        normdist = 1 / normdist;
        #pragma omp parallel for simd
        for (long p = 0; p < n; p++)
          distUnfilteredMatrix[p + j * n] = distUnfilteredMatrix[p + j * n] * normdist;
      }

      #pragma omp parallel for simd
      for (long p = 0; p < n; p++)
        distMatrix[p + (j - 1) * n] = distUnfilteredMatrix[p + j * n];
      correct_pivots++;
      endTimerPart = std::chrono::high_resolution_clock::now();
      elt = endTimerPart - startInnerTimer;
      eltOrth[t] += elt.count();
    }

    if(correct_pivots < numSources)
    {
      std::cerr<<"Number of Pivots obtained: "<<correct_pivots<<" inadequate, increase the Pivots Limits: "<<pivotsLimits<<std::endl;
      exit(0);
    }

    startTimerPartTemp = std::chrono::high_resolution_clock::now();
    /* Multiplication of Laplacians with untransposed distance matrix S */
    startInnerTimer = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int k = 0; k < numSources; k++)
    {
     for (long i = 0; i < n; i++)
     {
        this->LSmatrix[i + k * n] = degrees[i] * distMatrix[i + k * n];
        for (unsigned int j = rowPtrs[i]; j < rowPtrs[i + 1]; j++)
        {
          unsigned int v = this->adj[j];
          this->LSmatrix[i + k * n] -= weights[j] * distMatrix[v + k * n];
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
  double totalDistanceTime = 0, totalHDE = 0, totalOrth = 0,
         totalMMult = 0, meanDistanceTime, meanHDE, meanOrth, meanMMult;
  for (int t = 0; t < tries; t++)
  {
    totalDistanceTime += eltDistance[t];
    totalHDE += eltHDE[t];
    totalOrth += eltOrth[t];
    totalMMult += eltMMult[t];
  }
  meanDistanceTime = totalDistanceTime / tries;
  meanHDE = totalHDE / tries;
  meanOrth = totalOrth / tries;
  meanMMult = totalMMult / tries;
  totalDistanceTime = totalHDE = totalOrth = totalMMult = 0;

  std::cout << "**** Means ******" << std::endl;
  std::cout << "HDE mean time " << meanHDE << " s." << std::endl;
  std::cout << "SSSP mean time " << meanDistanceTime << " s." << std::endl;
  std::cout << "DOrthogonalize mean time " << meanOrth << " s." << std::endl;
  std::cout << "Matrix Multiply Mean time " << meanMMult << " s." << std::endl;

  #ifndef INTEL_MKL
  free(degrees);
  #else
  mkl_free(degrees);
  #endif
  return 0;
}