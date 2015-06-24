#include "../common/common.h"
#include <stdio.h>

/*
 * This example purposefully introduces an invalid memory access on the GPU to
 * illustrate the use of cuda-gdb.
 */

#define N   1025
#define M   12

__device__ int foo(int row, int col)
{
    return (2 * row);
}

__global__ void kernel(int **arr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i;

    /*
     * Iterate over each row in parallel and column sequentially, assigning a
     * value decided by foo.
     */
    for ( ; tid < N; tid++)
    {
        for (i = 0; i < M; i++)
        {
            arr[tid][i] = foo(tid, i);
        }
    }
}

int main(int argc, char **argv)
{
    int i;
    // Host representation of a 2D matrix
    int **h_matrix;
    // A host array of device pointers to the matrix rows on the device
    int **d_ptrs;
    // A device array of device pointers, filled from d_ptrs
    int **d_matrix;

    h_matrix = (int **)malloc(N * sizeof(int *));
    d_ptrs = (int **)malloc(N * sizeof(int *));
    CHECK(cudaMalloc((void **)&d_matrix, N * sizeof(int *)));
    CHECK(cudaMemset(d_matrix, 0x00, N * sizeof(int *)));

    // Allocate rows on the host and device
    for (i = 0; i < N; i++)
    {
        h_matrix[i] = (int *)malloc(M * sizeof(int));
        CHECK(cudaMalloc((void **)&d_ptrs[i], M * sizeof(int)));
        CHECK(cudaMemset(d_ptrs[i], 0x00, M * sizeof(int)));
    }

    CHECK(cudaMemcpy(d_matrix, d_ptrs, N * sizeof(int *),
                    cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = 1024;
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix);

    // Copy rows back
    for (i = 0; i < N; i++)
    {
        CHECK(cudaMemcpy(h_matrix[i], d_ptrs[i], M * sizeof(int),
                        cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_ptrs[i]));
        free(h_matrix[i]);
    }

    CHECK(cudaFree(d_matrix));
    free(h_matrix);

    return 0;
}
