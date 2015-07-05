#include "../common/common.h"
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

/*
 * A simple example of a multi-GPU CUDA application implementing a vector sum.
 * Note that all communication and computation is done asynchronously in order
 * to overlap computation across multiple devices, and that this requires
 * allocating page-locked host memory associated with a specific device.
 */

__global__ void iKernel(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                    gpuRef[i], i);
            break;
        }
    }
}

void initialData(float * const ip, int const  size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)rand() / (float)RAND_MAX;
    }
}

void sumOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv)
{
    int ngpus;
    cudaEvent_t start, end;

    printf("> starting %s", argv[0]);

    CHECK(cudaGetDeviceCount(&ngpus));
    printf(" CUDA-capable devices: %i\n", ngpus);

    int ishift = 24;

    if (argc > 2) ishift = atoi(argv[2]);

    int size = 1 << ishift;

    if (argc > 1)
    {
        if (atoi(argv[1]) > ngpus)
        {
            fprintf(stderr, "Invalid number of GPUs specified: %d is greater "
                    "than the total number of GPUs in this platform (%d)\n",
                    atoi(argv[1]), ngpus);
            exit(1);
        }

        ngpus  = atoi(argv[1]);
    }

    int    iSize  = size / ngpus;
    size_t iBytes = iSize * sizeof(float);

    printf("> total array size %d M, using %d devices with each device "
            "handling %d M\n", size / 1024 / 1024, ngpus, iSize / 1024 / 1024);

    // allocat device emory
    float **d_A = (float **)malloc(sizeof(float *) * ngpus);
    float **d_B = (float **)malloc(sizeof(float *) * ngpus);
    float **d_C = (float **)malloc(sizeof(float *) * ngpus);

    float **h_A = (float **)malloc(sizeof(float *) * ngpus);
    float **h_B = (float **)malloc(sizeof(float *) * ngpus);
    float **hostRef = (float **)malloc(sizeof(float *) * ngpus);
    float **gpuRef = (float **)malloc(sizeof(float *) * ngpus);
    cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * ngpus);

    for (int i = 0; i < ngpus; i++)
    {
        // set current device
        CHECK(cudaSetDevice(i));

        // allocate device memory
        CHECK(cudaMalloc((void **) &d_A[i], iBytes));
        CHECK(cudaMalloc((void **) &d_B[i], iBytes));
        CHECK(cudaMalloc((void **) &d_C[i], iBytes));

        // allocate page locked host memory for asynchronous data transfer
        CHECK(cudaMallocHost((void **) &h_A[i],     iBytes));
        CHECK(cudaMallocHost((void **) &h_B[i],     iBytes));
        CHECK(cudaMallocHost((void **) &hostRef[i], iBytes));
        CHECK(cudaMallocHost((void **) &gpuRef[i],  iBytes));

        // create streams for timing and synchronizing
        CHECK(cudaStreamCreate(&stream[i]));
    }

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));

    dim3 block (512);
    dim3 grid  ((iSize + block.x - 1) / block.x);

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        initialData(h_A[i], iSize);
        initialData(h_B[i], iSize);
    }

    // record start time
    double iStart = seconds();
    CHECK(cudaEventRecord(start));

    // distributing the workload across multiple devices
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        CHECK(cudaMemcpyAsync(d_A[i], h_A[i], iBytes, cudaMemcpyHostToDevice,
                              stream[i]));
        CHECK(cudaMemcpyAsync(d_B[i], h_B[i], iBytes, cudaMemcpyHostToDevice,
                              stream[i]));

        iKernel<<<grid, block, 0, stream[i]>>>(d_A[i], d_B[i], d_C[i], iSize);

        CHECK(cudaMemcpyAsync(gpuRef[i], d_C[i], iBytes, cudaMemcpyDeviceToHost,
                              stream[i]));
    }

    CHECK(cudaEventRecord(end));

    // synchronize streams
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamSynchronize(stream[i]));
    }

    // calculate the elapsed time in seconds
    double iElaps = seconds() - iStart;
    float eventsElapsed;
    CHECK(cudaEventElapsedTime(&eventsElapsed, start, end));
    printf("%d GPU timer elapsed: %8.2fms \n", ngpus, iElaps * 1000.0);
    printf("%d GPU events elapsed: %8.2fms\n", ngpus, eventsElapsed);

    // check results
    for (int i = 0; i < ngpus; i++)
    {
        //Set device
        CHECK(cudaSetDevice(i));
        sumOnHost(h_A[i], h_B[i], hostRef[i], iSize);
        checkResult(hostRef[i], gpuRef[i], iSize);
    }

    // Cleanup and shutdown
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaFree(d_A[i]));
        CHECK(cudaFree(d_B[i]));
        CHECK(cudaFree(d_C[i]));

        CHECK(cudaFreeHost(h_A[i]));
        CHECK(cudaFreeHost(h_B[i]));
        CHECK(cudaFreeHost(hostRef[i]));

        CHECK(cudaFreeHost(gpuRef[i]));
        CHECK(cudaStreamDestroy(stream[i]));

        CHECK(cudaDeviceReset());
    }

    free(d_A);
    free(d_B);
    free(d_C);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    free(stream);

    return EXIT_SUCCESS;
}
