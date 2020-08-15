#ifndef _PIVOT_MDS_H_
#define _PIVOT_MDS_H_

#include "DrawingBase.h"
#include "omp.h"
#ifdef INTEL_MKL
#include "mkl.h"
#endif

class PivotMDS : public DrawingBase
{
    private:
    double *matrixC;
    void doubleCentering(double *);
    void cTransCmatrixComputation(double *);

    public:
    PivotMDS(unsigned int *rowPtrs, unsigned int *adj, int32_t *weights, long n, long m, int s, int r, int numCoord);
    ~PivotMDS();

    virtual int run(int tries, int alpha, int beta, int32_t delta);
};

#endif