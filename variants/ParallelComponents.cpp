/** @file  ParallelComponents.cpp
 * 
 *  @brief Parallelized routines
 *
*/
#include "ParallelComponents.h"
#include <stdlib.h> 
#include <iostream>

ParallelComponents::ParallelComponents()
{}

double ParallelComponents::normL2(long n, double *vec)
{
  double norm2 = 0;
  #pragma omp parallel for reduction(+:norm2)
  for(long i=0; i < n; i++) 
    norm2 += pow(vec[i],2);
  return sqrt(norm2);
}

double ParallelComponents::dotProd(double *x, double *y, int n)
{
  double result = 0;
  #pragma omp parallel for reduction(+: result)
  for (int i = 0; i < n; i++)
    result += x[i] * y[i];
  return result;
}

double normL2Seq(long n, double *vec)
{
  double norm2 = 0;
  #pragma omp parallel for reduction(+:norm2)
  for(long i=0; i < n; i++) 
    norm2 += pow(vec[i],2);
  return sqrt(norm2);
}
double prodSeq(double *x, double *y, int n)
{
  double result = 0;
  #pragma omp parallel for reduction(+:result)
  for(int i = 0; i < n; i++)
    result += x[i] * y[i];
  return result;
}

int ParallelComponents::eigenValuePowerMethod(
  double *K,
  double *eValues,
  long n,
  int s)
{
  const double EPSILON = 1 - 1e-3;
  double *eVecs = new double[n*s];
  double *tmpOld = new double[n*s];
  srand(1);
  for(int i=0; i < s; i++)
  {
    for(int j=0; j < n; j++)
    {
      eVecs[j+i*n] = ((double) rand()) / RAND_MAX;
    }
  }
  const int p = n;
  double r = 0;
  for (int i = 0; i < s; i++) 
  {
    eValues[i] = normL2Seq(n, eVecs + i*n);
    for(int j=0; j < n; j++)
      eVecs[j+i*n] /= eValues[i];
  }
  int countIterations = 0;
  while (r < EPSILON) 
  {
    countIterations++;
    if (std::isnan(r) || std::isinf(r)) 
    {
      std::cerr<<"Error in power iteration"<<std::endl;
      return 1;
    }

    for (int i = 0; i < s; i++) 
    {
      for (int j = 0; j < p; j++) 
      {
        tmpOld[j+i*n] = eVecs[j+i*n];
        eVecs[j+i*n] = 0;
      }
    }
    for (int i = 0; i < s; i++) 
    {
      for (int j = 0; j < p; j++) 
      {
        for (int k = 0; k < p; k++) 
        {
          eVecs[k+i*p] += K[k+j*p] * tmpOld[j+i*p];
        }
      }
    }

    for (int i = 0; i < s; i++) 
    {
      for (int j = 0; j < i; j++) 
      {
        double fac = prodSeq(eVecs+j*p, eVecs+i*p, p) / prodSeq(eVecs+j*p, eVecs+j*p, p);
        for (int k = 0; k < p; k++) 
        {
          eVecs[k+i*p] -= fac * eVecs[k+j*p];
        }
      } 
    }
    for (int i = 0; i < s; i++) 
    {
      eValues[i] = normL2(p, eVecs+i*n);
      for(int j=0; j < n; j++)
      {
        eVecs[j+i*n] /= eValues[i];
      }
    }

    r = 1;
    for (int i = 0; i < s; i++) 
    {
      double tmp = prodSeq(eVecs+i*n, tmpOld+i*n, n);
      if (tmp < 0) 
        tmp *= -1;
      if(tmp < r)
        r = tmp;
    }
  }
  int start = 0, end = s-1;
  while (start < end) 
  { 
    double temp = eValues[start];  
    eValues[start] = eValues[end]; 
    eValues[end] = temp; 
    start++; 
    end--; 
  }

  bool signFlag = false;
  for(int i=0; i < s; i++)
  {
    if(eVecs[(s-i-1)*n] < 0) signFlag = true;
    for(int j=0; j < n; j++)
    {
      if(signFlag)
        K[j+(s-i-1)*n] = -eVecs[j+i*n];
      else
        K[j+(s-i-1)*n] = eVecs[j+i*n];
    }
    signFlag = false;
  }
  delete [] eVecs;
  delete [] tmpOld;
  return 0;
}
