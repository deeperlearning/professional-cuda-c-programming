#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

/*
 * This example uses the cuRAND host and device API to replace the system rand()
 * call by pre-generating large chunks of random numbers before fetching one at
 * at time. If there are no unused random numbers left, a new batch is generated
 * synchronously.
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
    static curandState *states = NULL;
    static float *dRand = NULL;
    static float *hRand = NULL;
    static int dRand_length = 1000000;
    static int dRand_used = 1000000;

    int threads_per_block = 256;
    int blocks_per_grid = 30;

    if (dRand == NULL)
    {
        /*
         * If the cuRAND state hasn't been initialized yet, pre-allocate memory
         * to store the generated random values in as well as the cuRAND device
         * state objects.
         */
        CHECK(cudaMalloc((void **)&dRand, sizeof(float) * dRand_length));
        CHECK(cudaMalloc((void **)&states, sizeof(curandState) *
                        threads_per_block * blocks_per_grid));
        hRand = (float *)malloc(sizeof(float) * dRand_length);
        // Initialize states on the device
        initialize_state<<<blocks_per_grid, threads_per_block>>>(states);
    }

    if (dRand_used == dRand_length)
    {
        /*
         * If all pre-generated random numbers have been consumed, regenerate a
         * new batch.
         */
        refill_randoms<<<blocks_per_grid, threads_per_block>>>(dRand,
                dRand_length, states);
        CHECK(cudaMemcpy(hRand, dRand, sizeof(float) * dRand_length,
                        cudaMemcpyDeviceToHost));
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
    static float *dRand = NULL;
    static float *hRand = NULL;
    curandGenerator_t randGen;
    static int dRand_length = 1000000;
    static int dRand_used = 1000000;

    if (dRand == NULL)
    {
        /*
         * If the cuRAND state hasn't been initialized yet, construct a cuRAND
         * host generator and pre-allocate memory to store the generated random
         * values in.
         */
        CHECK_CURAND(curandCreateGenerator(&randGen,
                                           CURAND_RNG_PSEUDO_DEFAULT));
        CHECK(cudaMalloc((void **)&dRand, sizeof(float) * dRand_length));
        hRand = (float *)malloc(sizeof(float) * dRand_length);
    }

    if (dRand_used == dRand_length)
    {
        /*
         * If all pre-generated random numbers have been consumed, regenerate a
         * new batch using curandGenerateUniform.
         */
        CHECK_CURAND(curandGenerateUniform(randGen, dRand, dRand_length));
        CHECK(cudaMemcpy(hRand, dRand, sizeof(float) * dRand_length,
                        cudaMemcpyDeviceToHost));
        dRand_used = 0;
    }

    // Return the next pre-generated random number
    return hRand[dRand_used++];
}

/*
 * A reference implementation that uses system rand().
 */
float host_rand()
{
    return (float)rand() / (float)RAND_MAX;
}

int main(int argc, char **argv)
{
    int i;
    int N = 8388608;

    /*
     * Allocate N random numbers from each of the random number generation
     * functions implemented.
     */
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
