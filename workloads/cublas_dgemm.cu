/**
 * @file cublas_dgemm.cu
 *
 * @brief This program creates n*n matrices 
 * in managed memory and performs matrix-matrix
 * multiplication using a single cuBLAS operation.
 *
 * @author Rui Pan
 * Contact: rpan33@wisc.edu
 *
 */

#include <cuda.h>
#include <iostream>
#include <random>
#include <time.h>
#include "mmul.h"

using namespace std;

/**This program creates n*n matrices 
 * in managed memory and performs matrix-matrix
 * multiplication using a single cuBLAS operation.
 */
int main(int argc, char *argv[])
{
    time_t mytime = time(NULL);
    char * time_str = ctime(&mytime);
    time_str[strlen(time_str)-1] = '\0';
    printf("Started program at : %s\n", time_str);

    // parse input
    int n = atol(argv[1]);
    int n_tests = atol(argv[2]);

    // set up the random float generator
    std::mt19937 generator(1234);
    uniform_real_distribution<double> dist(-1.0, 1.0);

    double *A, *B, *C;
    // allocate memory using managed memory
    cudaMallocManaged((void **)&A, n * n * sizeof(double));
    cudaMallocManaged((void **)&B, n * n * sizeof(double));
    cudaMallocManaged((void **)&C, n * n * sizeof(double));

    // fill in the host array with random floats in [-1, 1]
    for (int i = 0; i < n * n; i++)
    {
        A[i] = dist(generator);
        B[i] = dist(generator);
        C[i] = dist(generator);
    }

    // initialize the CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // create the cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    mytime = time(NULL);
    time_str = ctime(&mytime);
    time_str[strlen(time_str)-1] = '\0';
    printf("Started running at : %s\n", time_str);

    // call reduce() to sum all the elements in the input array
    cudaEventRecord(start);
    for (int i = 0; i < n_tests; i++) // invoke for n_tests times to reduce randomness
    {
        if (i % 10000 == 0) {
            mytime = time(NULL);
            time_str = ctime(&mytime);
            time_str[strlen(time_str)-1] = '\0';
            printf("Running test %d at : %s\n", i, time_str);
        }
        mmul(handle, A, B, C, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; // Get the elapsed time in milliseconds
    cudaEventElapsedTime(&ms, start, stop);
    // ms /= n_tests; // get the average

    printf("JCT: %f\n", ms/(float)1000); // print time taken to perform the multiplication in ms

    mytime = time(NULL);
    time_str = ctime(&mytime);
    time_str[strlen(time_str)-1] = '\0';
    printf("Finished running at : %s\n", time_str);

    // free stuff
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    cublasDestroy(handle);

    return 0;
}
