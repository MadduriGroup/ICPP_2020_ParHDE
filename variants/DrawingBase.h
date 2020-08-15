#ifndef _DRAWING_BASE_H_
#define _DRAWING_BASE_H_
#include <stdlib.h>
#include <limits.h>
#include "ParallelComponents.h"

enum EigenChoice
{
  First,
  Largest
};

class DrawingBase 
{
  private:
  void mkl_eigen_value_vector_comp_largest(double *matrix, long N, double *eigenVectors);
  void mkl_eigen_value_vector_comp_first(double *matrix, long N, double *eigenVectors);
    
  protected:
  unsigned int *rowPtrs;
  unsigned int *adj;
  int32_t *weights;
  long n;
  long m;
  int numSources;
  int numCoord;
  ParallelComponents parComp;
  double *eigenVectors;
  double *coordinates;
  const int ALIGN = 64;
  int doRand;
  void matrixTransMultiplication(double *A, double *B, double *C, long m, long k, long n);
  void distancePhaseComputation(double **, double *, int, int, int, int32_t);
  void DistanceAlgorithm(unsigned int *row, unsigned int *col, long source, long numNodes, long numEdges,
  double *dists, int32_t *weights = nullptr, int32_t delta = 1, int alpha = 15, int beta = 18);

  void computeCoordinates(double *);
  void seqBfs(unsigned int start, double *distCol);

  public:
  DrawingBase(unsigned int *, unsigned int *, int32_t *, long, long, int, int, int numCoord = 2);
  ~DrawingBase();

  double * getCoordinates();

  void compute_eigens(double *, int, double *, EigenChoice);
  virtual int run(int tries = 11, int alpha = 15, int beta = 18, int32_t delta = 1) = 0; // needs to be implemented
};

struct Compare
{
  double val;
  long index;
};
#ifdef _OPENMP
#pragma omp declare reduction(argmax           \
                              : struct Compare \
                              : omp_out = omp_in.val > omp_out.val ? omp_in : omp_out)
#endif


extern void DOBFS(unsigned int *row, unsigned int *col, long source, long numNodes, long numEdges, double *dists,
                  int alpha = 15,
                  int beta = 18);
extern double *DeltaStep(const unsigned int *row, unsigned int *col, int32_t *weights,
                         long source, long numNodes, long numEdges, double *dist,
                         int32_t delta);

#endif