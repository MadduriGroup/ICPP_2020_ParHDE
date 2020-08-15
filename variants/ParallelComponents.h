#ifndef _PARALLEL_COMPONENTS_H_
#define _PARALLEL_COMPONENTS_H_
#include "omp.h"
#include <cmath>

class ParallelComponents 
{
  public:

  ParallelComponents();
  double normL2(long n, double *vec);
  double dotProd(double *x, double *y, int n);
  int eigenValuePowerMethod(double *K, double *eValues, long n,int s);
};

#endif