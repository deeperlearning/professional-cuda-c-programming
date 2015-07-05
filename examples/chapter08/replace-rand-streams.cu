#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

/*
 * This example is a clone of replace-rand.cu that uses CUDA streams to overlap
 * the generation of random numbers using cuSPARSE with any host computation.
 */

/*
 * initialize_state initializes cuRAND device state
 */
__global__ void initialize_state(curandState *states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(9384, tid, 0, states + tid);
}

/*
 * refill_randoms uses the cuRAND device API to generate N random values using
 * the states passed to the kernel.
 */
__global__ void refill_randoms(float *dRand, int N, curandState *states)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;
    curandState *state = states + tid;

    for (i = tid; i < N; i += nthreads)
    {
        dRand[i] = curand_uniform(state);
    }
}

/*
 * An implementation of rand() that uses the cuRAND device API.
 */
float cuda_device_rand()
{
    static cudaStream_t stream = 0;
    static curandState *states = NULL;
    static float *dRand = NULL;
    static float *hRand = NULL;
    static int dRand_length = 1000000;
    static int dRand_used = dRand_length;

    int threads_per_block = 256;
    int blocks_per_grid = 30;

    if (dRand == NULL)
    {
        /*
         * If the cuRAND state hasn't been initialized yet, create a CUDA stream
         * to execute operations in, pre-allocate device memory to store the
         * generated random values in, and asynchronously launch a
         * refill_randoms kernel to begin generating random numbers.
         */
        CHECK(cudaStreamCreate(&stream));
        CHECK(cudaMalloc((void **)&dRand, sizeof(float) * dRand_length));
        CHECK(cudaMalloc((void **)&states, sizeof(curandState) *
                        threads_per_block * blocks_per_grid));
        hRand = (float *)malloc(sizeof(float) * dRand_length);
        initialize_state<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            states);
        refill_randoms<<<blocks_per_grid, threads_per_block>>>(dRand,
                dRand_length, states);
    }

    if (dRand_used == dRand_length)
    {
        /*
         * If all pre-generated random numbers have been consumed, wait for the
         * last launch of refill_randoms to complete, transfer those newly
         * generated random numbers back, and launch another batch random number
         * generation kernel asynchronously.
         */
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(hRand, dRand, sizeof(float) * dRand_length,
                    cudaMemcpyDeviceToHost));
        refill_randoms<<<blocks_per_grid, threads_per_block, 0, stream>>>(dRand,
                dRand_length, states);
        dRand_used = 0;
    }

    // Return the next pre-generated random number
    return hRand[dRand_used++];
}

/*
 * An implementation of rand() that uses the cuRAND host API.
 */
float cuda_host_rand()
{
    static cudaStream_t stream = 0;
    static float *dRand = NULL;
    static float *hRand = NULL;
    curandGenerator_t randGen;
    static int dRand_length = 1000000;
    static int dRand_used = 1000000;

    if (dRand == NULL)
    {
        /*
         * If the cuRAND state hasn't been initialized yet, construct a cuRAND
         * generator and configure it to use a CUDA stream. Pre-allocate device
         * memory to store the output random numbers and asynchronously launch
         * curandGenerateUniform. Because curandGenerateUniform uses the randGen
         * handle, it will execute in the set stream.
         */
        CHECK_CURAND(curandCreateGenerator(&randGen,
                                           CURAND_RNG_PSEUDO_DEFAULT));
        CHECK(cudaStreamCreate(&stream));
        CHECK_CURAND(curandSetStream(randGen, stream));

        CHECK(cudaMalloc((void **)&dRand, sizeof(float) * dRand_length));
        hRand = (float *)malloc(sizeof(float) * dRand_length);
        CHECK_CURAND(curandGenerateUniform(randGen, dRand, dRand_length));
    }

    if (dRand_used == dRand_length)
    {
        /*
         * If all pre-generated random numbers have been consumed, wait for the
         * last asynchronous curandGenerateUniform to complex, transfer the new
         * batch of random numbers back to the host, and relaunch
         * curandGenerateUniform.
         */
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(hRand, dRand, sizeof(float) * dRand_length,
                        cudaMemcpyDeviceToHost));
        CHECK_CURAND(curandGenerateUniform(randGen, dRand, dRand_length));
        dRand_used = 0;
    }

    // Return the next pre-generated random number
    return hRand[dRand_used++];
}

float host_rand()
{
    return (float)rand() / (float)RAND_MAX;
}

int main(int argc, char **argv)
{
    int i;
    int N = 8388608;

    for (i = 0; i < N; i++)
    {
        float h = host_rand();
        float d = cuda_host_rand();
        float dd = cuda_device_rand();
        printf("%2.4f %2.4f %2.4f\n", h, d, dd);
        getchar();
    }

    return 0;
}
