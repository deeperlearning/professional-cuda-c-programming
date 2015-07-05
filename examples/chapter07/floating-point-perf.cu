#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * This example illustrates the performance cost of using double-precision
 * floating point values, relative to single-precision floating-point values.
 *
 * The core computational kernel executes a number of mathematical operations on
 * an input value. This example includes a kernel for both single- and
 * double-precision floating-point. Timing statements are placed around the copy
 * of inputs to the device, the copy of outputs from the device, and the
 * execution of the kernel. These timing statements enable comparison of
 * overhead from both communication and computation.
 **/

/**
 * The computational kernel for single-precision floating-point
 **/
__global__ void lots_of_float_compute(float *inputs, int N, size_t niters,
                                      float *outputs)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nthreads = gridDim.x * blockDim.x;

    for ( ; tid < N; tid += nthreads)
    {
        size_t iter;
        float val = inputs[tid];

        for (iter = 0; iter < niters; iter++)
        {
            val = (val + 5.0f) - 101.0f;
            val = (val / 3.0f) + 102.0f;
            val = (val + 1.07f) - 103.0f;
            val = (val / 1.037f) + 104.0f;
            val = (val + 3.00f) - 105.0f;
            val = (val / 0.22f) + 106.0f;
        }

        outputs[tid] = val;
    }
}

/**
 * The computational kernel for double-precision floating-point
 **/
__global__ void lots_of_double_compute(double *inputs, int N, size_t niters,
                                       double *outputs)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nthreads = gridDim.x * blockDim.x;

    for ( ; tid < N; tid += nthreads)
    {
        size_t iter;
        double val = inputs[tid];

        for (iter = 0; iter < niters; iter++)
        {
            val = (val + 5.0) - 101.0;
            val = (val / 3.0) + 102.0;
            val = (val + 1.07) - 103.0;
            val = (val / 1.037) + 104.0;
            val = (val + 3.00) - 105.0;
            val = (val / 0.22) + 106.0;
        }

        outputs[tid] = val;
    }
}

/**
 * Runs a full test of single-precision floating-point, including transferring
 * inputs to the device, running the single-precision kernel, and copying
 * outputs back.
 **/
static void run_float_test(size_t N, int niters, int blocksPerGrid,
                           int threadsPerBlock, double *toDeviceTime,
                           double *kernelTime, double *fromDeviceTime,
                           float *sample, int sampleLength)
{
    int i;
    float *h_floatInputs, *h_floatOutputs;
    float *d_floatInputs, *d_floatOutputs;

    h_floatInputs = (float *)malloc(sizeof(float) * N);
    h_floatOutputs = (float *)malloc(sizeof(float) * N);
    CHECK(cudaMalloc((void **)&d_floatInputs, sizeof(float) * N));
    CHECK(cudaMalloc((void **)&d_floatOutputs, sizeof(float) * N));

    for (i = 0; i < N; i++)
    {
        h_floatInputs[i] = (float)i;
    }

    double toDeviceStart = seconds();
    CHECK(cudaMemcpy(d_floatInputs, h_floatInputs, sizeof(float) * N,
                     cudaMemcpyHostToDevice));
    *toDeviceTime = seconds() - toDeviceStart;

    double kernelStart = seconds();
    lots_of_float_compute<<<blocksPerGrid, threadsPerBlock>>>(d_floatInputs,
            N, niters, d_floatOutputs);
    CHECK(cudaDeviceSynchronize());
    *kernelTime = seconds() - kernelStart;

    double fromDeviceStart = seconds();
    CHECK(cudaMemcpy(h_floatOutputs, d_floatOutputs, sizeof(float) * N,
                     cudaMemcpyDeviceToHost));
    *fromDeviceTime = seconds() - fromDeviceStart;

    for (i = 0; i < sampleLength; i++)
    {
        sample[i] = h_floatOutputs[i];
    }

    CHECK(cudaFree(d_floatInputs));
    CHECK(cudaFree(d_floatOutputs));
    free(h_floatInputs);
    free(h_floatOutputs);
}

/**
 * Runs a full test of double-precision floating-point, including transferring
 * inputs to the device, running the single-precision kernel, and copying
 * outputs back.
 **/
static void run_double_test(size_t N, int niters, int blocksPerGrid,
                            int threadsPerBlock, double *toDeviceTime,
                            double *kernelTime, double *fromDeviceTime,
                            double *sample, int sampleLength)
{
    int i;
    double *h_doubleInputs, *h_doubleOutputs;
    double *d_doubleInputs, *d_doubleOutputs;

    h_doubleInputs = (double *)malloc(sizeof(double) * N);
    h_doubleOutputs = (double *)malloc(sizeof(double) * N);
    CHECK(cudaMalloc((void **)&d_doubleInputs, sizeof(double) * N));
    CHECK(cudaMalloc((void **)&d_doubleOutputs, sizeof(double) * N));

    for (i = 0; i < N; i++)
    {
        h_doubleInputs[i] = (double)i;
    }

    double toDeviceStart = seconds();
    CHECK(cudaMemcpy(d_doubleInputs, h_doubleInputs, sizeof(double) * N,
                     cudaMemcpyHostToDevice));
    *toDeviceTime = seconds() - toDeviceStart;

    double kernelStart = seconds();
    lots_of_double_compute<<<blocksPerGrid, threadsPerBlock>>>(d_doubleInputs,
            N, niters, d_doubleOutputs);
    CHECK(cudaDeviceSynchronize());
    *kernelTime = seconds() - kernelStart;

    double fromDeviceStart = seconds();
    CHECK(cudaMemcpy(h_doubleOutputs, d_doubleOutputs, sizeof(double) * N,
                     cudaMemcpyDeviceToHost));
    *fromDeviceTime = seconds() - fromDeviceStart;

    for (i = 0; i < sampleLength; i++)
    {
        sample[i] = h_doubleOutputs[i];
    }

    CHECK(cudaFree(d_doubleInputs));
    CHECK(cudaFree(d_doubleOutputs));
    free(h_doubleInputs);
    free(h_doubleOutputs);
}

int main(int argc, char **argv)
{
    int i;
    double meanFloatToDeviceTime, meanFloatKernelTime, meanFloatFromDeviceTime;
    double meanDoubleToDeviceTime, meanDoubleKernelTime,
           meanDoubleFromDeviceTime;
    struct cudaDeviceProp deviceProperties;
    size_t totalMem, freeMem;
    float *floatSample;
    double *doubleSample;
    int sampleLength = 10;
    int nRuns = 5;
    int nKernelIters = 20;

    meanFloatToDeviceTime = meanFloatKernelTime = meanFloatFromDeviceTime = 0.0;
    meanDoubleToDeviceTime = meanDoubleKernelTime =
                                 meanDoubleFromDeviceTime = 0.0;

    CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

    size_t N = (freeMem * 0.9 / 2) / sizeof(double);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    if (blocksPerGrid > deviceProperties.maxGridSize[0])
    {
        blocksPerGrid = deviceProperties.maxGridSize[0];
    }

    printf("Running %d blocks with %d threads/block over %lu elements\n",
           blocksPerGrid, threadsPerBlock, N);

    floatSample = (float *)malloc(sizeof(float) * sampleLength);
    doubleSample = (double *)malloc(sizeof(double) * sampleLength);

    for (i = 0; i < nRuns; i++)
    {
        double toDeviceTime, kernelTime, fromDeviceTime;

        run_float_test(N, nKernelIters, blocksPerGrid, threadsPerBlock,
                       &toDeviceTime, &kernelTime, &fromDeviceTime,
                       floatSample, sampleLength);
        meanFloatToDeviceTime += toDeviceTime;
        meanFloatKernelTime += kernelTime;
        meanFloatFromDeviceTime += fromDeviceTime;

        run_double_test(N, nKernelIters, blocksPerGrid, threadsPerBlock,
                        &toDeviceTime, &kernelTime, &fromDeviceTime,
                        doubleSample, sampleLength);
        meanDoubleToDeviceTime += toDeviceTime;
        meanDoubleKernelTime += kernelTime;
        meanDoubleFromDeviceTime += fromDeviceTime;

        if (i == 0)
        {
            int j;
            printf("Input\tDiff Between Single- and Double-Precision\n");
            printf("------\t------\n");

            for (j = 0; j < sampleLength; j++)
            {
                printf("%d\t%.20e\n", j,
                       fabs(doubleSample[j] - (double)floatSample[j]));
            }

            printf("\n");
        }
    }

    meanFloatToDeviceTime /= nRuns;
    meanFloatKernelTime /= nRuns;
    meanFloatFromDeviceTime /= nRuns;
    meanDoubleToDeviceTime /= nRuns;
    meanDoubleKernelTime /= nRuns;
    meanDoubleFromDeviceTime /= nRuns;

    printf("For single-precision floating point, mean times for:\n");
    printf("  Copy to device:   %f s\n", meanFloatToDeviceTime);
    printf("  Kernel execution: %f s\n", meanFloatKernelTime);
    printf("  Copy from device: %f s\n", meanFloatFromDeviceTime);
    printf("For double-precision floating point, mean times for:\n");
    printf("  Copy to device:   %f s (%.2fx slower than single-precision)\n",
           meanDoubleToDeviceTime,
           meanDoubleToDeviceTime / meanFloatToDeviceTime);
    printf("  Kernel execution: %f s (%.2fx slower than single-precision)\n",
           meanDoubleKernelTime,
           meanDoubleKernelTime / meanFloatKernelTime);
    printf("  Copy from device: %f s (%.2fx slower than single-precision)\n",
           meanDoubleFromDeviceTime,
           meanDoubleFromDeviceTime / meanFloatFromDeviceTime);

    return 0;
}
