#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufftXt.h>

/*
 * An example usage of the Multi-GPU cuFFT XT library introduced in CUDA 6. This
 * example performs a 1D forward FFT across all devices detected in the system.
 */

/*
 * Create N fake samplings along the function cos(x). These samplings will be
 * stored as single-precision floating-point values.
 */
void generate_fake_samples(int N, float **out)
{
    int i;
    float *result = (float *)malloc(sizeof(float) * N);
    double delta = M_PI / 4.0;

    for (i = 0; i < N; i++)
    {
        result[i] = cos(i * delta);
    }

    *out = result;
}

/*
 * Convert a real-valued vector r of length Nto a complex-valued vector.
 */
void real_to_complex(float *r, cufftComplex **complx, int N)
{
    int i;
    (*complx) = (cufftComplex *)malloc(sizeof(cufftComplex) * N);

    for (i = 0; i < N; i++)
    {
        (*complx)[i].x = r[i];
        (*complx)[i].y = 0;
    }
}

/*
 * Retrieve device IDs for all CUDA devices in the current system.
 */
int getAllGpus(int **gpus)
{
    int i;
    int nGpus;

    CHECK(cudaGetDeviceCount(&nGpus));

    *gpus = (int *)malloc(sizeof(int) * nGpus);

    for (i = 0; i < nGpus; i++)
    {
        (*gpus)[i] = i;
    }

    return nGpus;
}

int main(int argc, char **argv)
{
    int i;
    int N = 1024;
    float *samples;
    cufftComplex *complexSamples;
    int *gpus;
    size_t *workSize;
    cufftHandle plan = 0;
    cudaLibXtDesc *dComplexSamples;

    int nGPUs = getAllGpus(&gpus);
    nGPUs = nGPUs > 2 ? 2 : nGPUs;
    workSize = (size_t *)malloc(sizeof(size_t) * nGPUs);

    // Setup the cuFFT Multi-GPU plan
    CHECK_CUFFT(cufftCreate(&plan));
    // CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
    CHECK_CUFFT(cufftXtSetGPUs(plan, 2, gpus));
    CHECK_CUFFT(cufftMakePlan1d(plan, N, CUFFT_C2C, 1, workSize));

    // Generate inputs
    generate_fake_samples(N, &samples);
    real_to_complex(samples, &complexSamples, N);
    cufftComplex *complexFreq = (cufftComplex *)malloc(
                                    sizeof(cufftComplex) * N);

    // Allocate memory across multiple GPUs and transfer the inputs into it
    CHECK_CUFFT(cufftXtMalloc(plan, &dComplexSamples, CUFFT_XT_FORMAT_INPLACE));
    CHECK_CUFFT(cufftXtMemcpy(plan, dComplexSamples, complexSamples,
                              CUFFT_COPY_HOST_TO_DEVICE));

    // Execute a complex-to-complex 1D FFT across multiple GPUs
    CHECK_CUFFT(cufftXtExecDescriptorC2C(plan, dComplexSamples, dComplexSamples,
                                         CUFFT_FORWARD));

    // Retrieve the results from multiple GPUs into host memory
    CHECK_CUFFT(cufftXtMemcpy(plan, complexSamples, dComplexSamples,
                              CUFFT_COPY_DEVICE_TO_HOST));

    printf("Fourier Coefficients:\n");

    for (i = 0; i < 30; i++)
    {
        printf("  %d: (%2.4f, %2.4f)\n", i + 1, complexFreq[i].x,
               complexFreq[i].y);
    }

    free(gpus);
    free(samples);
    free(complexSamples);
    free(complexFreq);
    free(workSize);

    CHECK_CUFFT(cufftXtFree(dComplexSamples));
    CHECK_CUFFT(cufftDestroy(plan));

    return 0;
}
