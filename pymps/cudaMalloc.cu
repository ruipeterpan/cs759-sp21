/**
 * @file cudaMalloc.cu
 *
 * @brief This program uses cudaMalloc
 * to allocate memory on the GPU, 
 * which brings up an NVIDIA MPS server
 * if a control daemon is already spawned.
 *
 * @author Rui Pan
 * Contact: rpan33@wisc.edu
 *
 */

#include <cuda.h>

using namespace std;

int main(int argc, char *argv[])
{
    int *arr;
    cudaMalloc((void **)&arr, 1 * sizeof(int));
    cudaFree(arr);
    return 0;
}
