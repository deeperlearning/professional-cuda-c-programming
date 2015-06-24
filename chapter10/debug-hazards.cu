#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * This example illustrates different approaches to optimizing access to a
 * single shared variable by limiting conflicting, atomic operations on it.
 *
 * The first kernel, naive_reduction, simply performs an atomicAdd from every
 * thread on the same shared variable.
 *
 * simple_reduction first stores the values to be added together in shared
 * memory. Then, a single thread iterates over those values and computes a
 * partial sum. Finally, that partial sum is added to the global result using an
 * atomicAdd.
 *
 * parallel_reduction is the most complex example. It performs a parallel
 * reduction within each thread block. The partial result produced by that
 * local reduction is then added to the global result with an atomicAdd.
 *
 * The core of each of these kernels is wrapped in a loop to augment the amount
 * of work done and make timing the kernels at the millisecond granularity
 * feasible.
 **/

/**
 * This implementation makes use of shared memory and local reduction to improve
 * performance and decrease contention
 **/
__global__ void simple_reduction(int *shared_var, int *input_values, int N,
                                 int iters)
{
    __shared__ int local_mem[256];
    int iter, i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    int local_dim = blockDim.x;
    int minThreadInThisBlock = blockIdx.x * blockDim.x;
    int maxThreadInThisBlock = minThreadInThisBlock + (blockDim.x - 1);

    if (maxThreadInThisBlock >= N)
    {
        local_dim = N - minThreadInThisBlock;
    }

    for (iter = 0; iter < iters; iter++)
    {
        if (tid < N)
        {
            local_mem[local_tid] = input_values[tid];
        }

        // Required for correctness
        // __syncthreads();

        /*
         * Perform the local reduction across values written to shared memory
         * by threads in this thread block.
         */
        if (local_tid == 0)
        {
            int sum = 0;

            for (i = 0; i < local_dim; i++)
            {
                sum = sum + local_mem[i];
            }

            atomicAdd(shared_var, sum);
        }

        // Required for correctness
        // __syncthreads();
    }
}

int main(int argc, char **argv)
{
    int N = 20480;
    int block = 256;
    int device_iters = 3;
    int runs = 1;
    int i, true_value;
    int *d_shared_var, *d_input_values, *h_input_values;
    int h_sum;
    double mean_time = 0.0;

    CHECK(cudaMalloc((void **)&d_shared_var, sizeof(int)));
    CHECK(cudaMalloc((void **)&d_input_values, N * sizeof(int)));
    h_input_values = (int *)malloc(N * sizeof(int));

    for (i = 0; i < N; i++)
    {
        h_input_values[i] = i;
        true_value += i;
    }

    true_value *= device_iters;

    for (i = 0; i < runs; i++)
    {
        CHECK(cudaMemset(d_shared_var, 0x00, sizeof(int)));
        CHECK(cudaMemcpy(d_input_values, h_input_values, N * sizeof(int),
                         cudaMemcpyHostToDevice));
        double start = seconds();

        simple_reduction<<<N / block, block>>>(d_shared_var,
                d_input_values, N, device_iters);

        CHECK(cudaDeviceSynchronize());
        mean_time += seconds() - start;
        CHECK(cudaMemcpy(&h_sum, d_shared_var, sizeof(int),
                         cudaMemcpyDeviceToHost));

        if (h_sum != true_value)
        {
            fprintf(stderr, "Validation failure: expected %d, got %d\n",
                    true_value, h_sum);
            return 1;
        }
    }

    mean_time /= runs;

    printf("Mean execution time for reduction: %.4f ms\n",
           mean_time * 1000.0);

    return 0;
}
