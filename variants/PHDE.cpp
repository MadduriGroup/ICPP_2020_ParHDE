/** @file  PHDE.cpp
 * 
 *  @brief PHDE implementation.
 *
*/
#include "PHDE.h"
#include <cmath>
#include <chrono>
#include <iostream>

PHDE::PHDE(unsigned int *rowPtrs, unsigned int *adj, int32_t *weights, long n, long m, int numSources, int doRand, int numCoord)
: DrawingBase(rowPtrs, adj, weights, n, m, numSources, doRand, numCoord)
{
  #ifndef INTEL_MKL
  this->matrixC = (double *)malloc(sizeof(double) * n * numSources);
  #else
  this->matrixC = (double *)mkl_malloc(sizeof(double) * n * numSources, ALIGN);
  #endif
}

PHDE::~PHDE()
{
  #ifndef INTEL_MKL
  free(matrixC);
  #else
  mkl_free(matrixC);
  #endif
}

void PHDE::columnCentering(double *delta)
{
  for(int i=0; i < numSources; i++)
  {
    double columnSum = 0 ;
    #pragma omp parallel for reduction(+:columnSum) schedule(static)
    for(int j=0; j < n; j++)
    {
      columnSum += delta[j + i*n];
    }
    columnSum /= n;
    #pragma omp parallel for schedule(static)
    for(int j=0; j < n; j++)
    {
      this->matrixC[j + i*n] = delta[j + i*n] - columnSum;
    }
  }
}


int PHDE::run(int tries = 1, int alpha = 15, int beta = 18, int32_t delta = 1)
{
  std::cout<<"Performing PHDE with "<<tries<<" tries and pivots: "<<numSources<<std::endl;
  #pragma omp parallel
  {
    #pragma omp single
    printf("Number of threads launched : %d \n", omp_get_num_threads());
  }
  double *eltDistance = new double[tries], *eltColumnCentering = new double[tries],
         *eltMMult = new double[tries], *eltOthers = new double[tries];
  auto startInnerTimer = std::chrono::high_resolution_clock::now();
  auto endTimerPart = std::chrono::high_resolution_clock::now();
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
    columnCentering(Distances);
    endTimerPart = std::chrono::high_resolution_clock::now();
    elt = endTimerPart - startInnerTimer;
    eltColumnCentering[trial] += elt.count();

    // C'C matrix multiplication
    startInnerTimer = std::chrono::high_resolution_clock::now();
    cTransCmatrixComputation(cTransC);
    endTimerPart = std::chrono::high_resolution_clock::now();
    elt = endTimerPart - startInnerTimer;
    eltMMult[trial] += elt.count();
   
    // eigen value-vector computation
    startInnerTimer = std::chrono::high_resolution_clock::now();
    compute_eigens(cTransC, numSources, this->eigenVectors, Largest);
    this->computeCoordinates(this->matrixC);
   
    endTimerPart = std::chrono::high_resolution_clock::now();
    elt = endTimerPart - startInnerTimer;
    eltOthers[trial] += elt.count();
  }

  #ifndef INTEL_MKL
  free(Distances);
  free(cTransC);
  #else
  mkl_free(Distances);
  mkl_free(cTransC);
  #endif
  double totalBfs = 0, totalHDE = 0, totalOthers = 0, totalMMult = 0, 
         meanBFS, meanTotal, meanMMult, totalCCenter = 0,
         meanCCenter;
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
  totalBfs = totalHDE = totalOthers = totalMMult = totalCCenter = 0;

  std::cout << "**** PHDE Means ******" << std::endl;
  std::cout << "Total mean time " << meanTotal << " s." << std::endl;
  std::cout << "BFS mean time " << meanBFS << " s." << std::endl;
  std::cout << "Matrix Multiply Mean time " << meanMMult << " s." << std::endl;
  std::cout << "ColumnCentering Mean time " << meanCCenter << " s." << std::endl;
  delete [] eltDistance;
  delete [] eltColumnCentering;
  delete [] eltMMult;
  delete [] eltOthers;
  return 0;
}

void PHDE::cTransCmatrixComputation(double *cTransC)
{
  matrixTransMultiplication(this->matrixC, this->matrixC, cTransC, numSources, n, numSources);
}