#ifndef _P_HDE_H_
#define _P_HDE_H_

#include "DrawingBase.h"
#include "omp.h"
#ifdef INTEL_MKL
#include "mkl.h"
#endif

class PHDE : public DrawingBase
{
    private:
    double *matrixC;
    void columnCentering(double *);
    void cTransCmatrixComputation(double *);

    public:
    PHDE(unsigned int *rowPtrs, unsigned int *adj, int32_t *weights, long n, long m, int s, int r, int numCoord);
    ~PHDE();

    virtual int run(int tries, int alpha, int beta, int32_t delta);
};

#endif