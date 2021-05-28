#include "mmul.h"

void mmul(cublasHandle_t handle, const double *A, const double *B, double *C, int n)
{
    const double alpha = (double)1.0;
    const double beta = (double)1.0;
    cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                A, n,
                B, n,
                &beta,
                C, n);
    cudaDeviceSynchronize();
    return;
}