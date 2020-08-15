#ifndef _W_PAR_HDE_H_
#define _W_PAR_HDE_H_

#include "DrawingBase.h"
#include "omp.h"
#ifdef INTEL_MKL
#include "mkl.h"
#endif

class WParHDE : public DrawingBase
{
  private:
  double *SLSmatrix;
  double *LSmatrix;
  void LSComputation(double *);
  void SLSComputation(double *);

  public:
  WParHDE(unsigned int *rowPtrs, unsigned int *adj, int32_t *weights, long n, long m, int s, int numCoord);
  ~WParHDE();

  virtual int run(int tries, int alpha, int beta, int32_t delta);
};

#endif
