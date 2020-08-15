/** @file  DrawingBase.cpp
 * 
 *  @brief Base Class for all drawing algorithms.
 *
*/
#include "DrawingBase.h"
#include <string.h>
#include <set>
#include <iostream>
#include <chrono>
#include <queue>
#ifdef INTEL_MKL
#include "mkl.h"
#endif

DrawingBase::DrawingBase(unsigned int *rowPtrs, unsigned int *adj, int32_t *weights, long n, long m, int numSources, int doRand, int numCoord)
    : rowPtrs(rowPtrs), adj(adj), weights(weights), n(n), m(m), numSources(numSources), numCoord(numCoord), doRand(doRand)
{
  #ifndef INTEL_MKL
  this->eigenVectors = (double *)malloc(sizeof(double) * numSources * numCoord);
  this->coordinates = (double *)malloc(sizeof(double) * n * numCoord);
  #else
  this->eigenVectors = (double *)mkl_malloc(sizeof(double) * numSources * numCoord, ALIGN);
  this->coordinates = (double *)mkl_malloc(sizeof(double) * n * numCoord, ALIGN);
  #endif
}

DrawingBase::~DrawingBase()
{
  #ifndef INTEL_MKL
  free(this->eigenVectors);
  free(this->coordinates);
  #else
  mkl_free(this->eigenVectors);
  mkl_free(this->coordinates);
  #endif
}

void DrawingBase::compute_eigens(double *matrix, int dimension, double *eigenVectors, EigenChoice c)
{
  switch(c)
  {
    case Largest:
      mkl_eigen_value_vector_comp_largest(matrix, dimension, eigenVectors);
      break;
    default:
      mkl_eigen_value_vector_comp_first(matrix, dimension, eigenVectors);
  }
}

void DrawingBase::mkl_eigen_value_vector_comp_largest(double *SLSMatrix, long N, double *eigenVectors)
{
  /* Locals */
  long numEigens = this->numCoord, n = N, lda = N, info;
  /* Local arrays */
  double *a = SLSMatrix;
  #ifndef INTEL_MKL
  double *w = (double *)malloc(N * sizeof(double));
  info = this->parComp.eigenValuePowerMethod(a, w, n, lda);
  #else
  double *w = (double *)mkl_malloc(N * sizeof(double), ALIGN);
  info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'L', n, a, lda, w);
  #endif
  /* Check for convergence */
  if (info > 0)
  {
    printf("The algorithm failed to compute eigenvalues.\n");
    exit(1);
  }
  for (long j = 0; j < numEigens; j++)
    for (long i = 0; i < lda; i++)
      eigenVectors[i + j * lda] = a[i + (lda-j-1) * lda];
  #ifdef INTEL_MKL
  mkl_free(w);
  #else
  free(w);
  #endif
}

void DrawingBase::mkl_eigen_value_vector_comp_first(double *SLSMatrix, long N, double *eigenVectors)
{
  /* Locals */
  long numEigens = this->numCoord, n = N, lda = N, info;
  /* Local arrays */
  double *a = SLSMatrix;
  #ifndef INTEL_MKL
  double *w = (double *)malloc(N * sizeof(double));
  info = this->parComp.eigenValuePowerMethod(a, w, n, lda);
  #else
  double *w = (double *)mkl_malloc(N * sizeof(double), ALIGN);
  info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'L', n, a, lda, w);
  #endif
  /* Check for convergence */
  if (info > 0)
  {
    printf("The algorithm failed to compute eigenvalues.\n");
    exit(1);
  }
  
  for (long j = 0; j < numEigens; j++)
    for (long i = 0; i < lda; i++)
        eigenVectors[i + j * lda] = a[i + j * lda];
  #ifdef INTEL_MKL
  mkl_free(w);
  #else
  free(w);
  #endif
}

void DrawingBase::DistanceAlgorithm(unsigned int *row, unsigned int *col, long source, long numNodes, long numEdges,
                                    double *dists, int32_t *weights, int32_t delta, int alpha, int beta)
{
  if (weights == nullptr) // performing BFS
    DOBFS(row, col, source, numNodes, numEdges, dists, alpha, beta = 18);
  else
    dists = DeltaStep(row, col, weights, source, numNodes, numEdges, dists, delta);
}

void DrawingBase::distancePhaseComputation(double **distMatrix, double *eltDistance, int trial,
                                         int alpha, int beta, int32_t delta)
{
  auto startOverallTimer = std::chrono::high_resolution_clock::now();
  auto endTimerPart = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elt;
  if(doRand==2)
  {
    double *min_dists = (double *)malloc(n * sizeof(double));
    long start_idx = 0;
    std::set<long> sources; // container to store unique start indices
    #pragma omp parallel for
    for (long p = 0; p < n; p++)
      min_dists[p] = INT_MAX;
    for (int run_count = 0; run_count < numSources; run_count++)
    {
      // numSources Distance computation
      DistanceAlgorithm(this->rowPtrs, this->adj, start_idx, n, m, ((*distMatrix) + run_count * n), this->weights);
      endTimerPart = std::chrono::high_resolution_clock::now();
      long old_idx = start_idx;
      struct Compare max_dist;
      max_dist.val = -1;
      max_dist.index = 0;
      #pragma omp parallel shared(max_dist)
      {
        #pragma omp for reduction(argmax : max_dist)
        for (long i = 0; i < n; i++)
        {
          if ((*distMatrix)[i + run_count * n] < min_dists[i])
            min_dists[i] = (*distMatrix)[i + run_count * n];

          if (min_dists[i] > max_dist.val)
          {
            max_dist.val = min_dists[i];
            max_dist.index = i;
          }
        }
      }
      start_idx = max_dist.index;
      if (run_count)
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
    free(min_dists);
  }
  else
  {
    unsigned *sources = new unsigned[numSources];
    for(int run_count = 0; run_count < numSources; run_count++)
      sources[run_count] = rand() % n;
    if(doRand==0) // rand pivots for small graphs
    {
      #pragma omp parallel for
      for (int run_count = 0; run_count < numSources; run_count++)
        this->seqBfs(sources[run_count-1], ((*distMatrix) + run_count * n));
    }
    else  // rand pivots for large graphs
    {
      for (int run_count = 0; run_count < numSources; run_count++)
        this->DistanceAlgorithm(rowPtrs, adj, sources[run_count-1], n, m, ((*distMatrix) + run_count * n));
    }
    delete [] sources;
  }
  endTimerPart = std::chrono::high_resolution_clock::now();
  elt = endTimerPart - startOverallTimer;
  eltDistance[trial] = elt.count();
}

double * DrawingBase::getCoordinates()
{
  for(int i=0; i < numCoord; i++)
  {
    double norm = 1 / parComp.normL2(n, this->coordinates+i*n);
    for(int j=0; j < n; j++)
      this->coordinates[j+i*n] *= norm;
  }
  return this->coordinates;
}

void DrawingBase::computeCoordinates(double *mat)
{
  #pragma omp parallel for
  for(int i=0; i < n; i++)
  {
    for(int k=0; k < numCoord; k++)
    {
      this->coordinates[i + k*n] = 0;
      for(int j=0 ; j < numSources; j++)
        this->coordinates[i + k*n] += mat[i + j*n] * this->eigenVectors[j + k*numSources];
     }
  }
}

void DrawingBase::matrixTransMultiplication(double *A, double *B, double *C, long mdim, long kdim, long ndim)
{
  /* A' is m*k so A is k*m, B is k*n, and C is m*n */
  #ifndef INTEL_MKL
  #pragma omp parallel for collapse(2) schedule(guided)
  for(int i=0; i < mdim; i++)
  {
    for(int k=0; k < ndim; k++)
    {
      double sum = 0;
      for(int j=0; j < kdim; j++)
      {
        sum += A[j + i*kdim] * B[j + k*kdim];
      }
      C[k + i*mdim] = sum;
    }
  }
  #else
    
  // lda = k, ldb = n, ldc = m
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
              mdim, ndim, kdim, 1, A, kdim, B, kdim, 0, C, mdim);
  #endif
}

void DrawingBase::seqBfs(unsigned int start, double *distCol) 
{
  unsigned int s = start;
  unsigned int *visited = (unsigned int *) malloc (sizeof(int) * n);
  memset(visited, 0, sizeof(unsigned int) * n);
  std::queue<unsigned int> Q;
  Q.push(s);
  visited[s] = 1;
  distCol[s] = 0;

  while(!Q.empty()) 
  {
    unsigned int h = Q.front();
    Q.pop();
    for (unsigned int j=this->rowPtrs[h]; j < this->rowPtrs[h+1]; j++) 
    {
      s = this->adj[j];
      if (!visited[s]) 
      {
        visited[s] = 1;
        Q.push(s);
        distCol[s] = distCol[h] + 1;
      }
    }
  }
  free(visited);
}