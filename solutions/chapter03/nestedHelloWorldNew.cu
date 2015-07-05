#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void nestedHelloWorld(int const iSize, int minSize, int iDepth)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d\n", iDepth, tid);

    // condition to stop recursive execution
    if (iSize == minSize) return;

    // reduce nthreads by half
    int nthreads = iSize >> 1;

    // thread 0 launches child grid recursively
    if(tid == 0 && nthreads > 0)
    {
        int blocks = (nthreads + blockDim.x - 1) / blockDim.x;
        nestedHelloWorld<<<blocks, blockDim.x>>>(nthreads, minSize, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char **argv)
{
    int igrid = 1;
    int blocksize = 8;

    if(argc > 1)
    {
        igrid = atoi(argv[1]);
    }

    if (argc > 2)
    {
        blocksize = atoi(argv[2]);
    }

    int size = igrid * blocksize;

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("size = %d\n", size);
    printf("igrid = %d\n", igrid);
    printf("%s Execution Configuration: grid %d block %d\n", argv[0], grid.x,
           block.x);

    nestedHelloWorld<<<grid, block>>>(size, grid.x, 0);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    return 0;
}
