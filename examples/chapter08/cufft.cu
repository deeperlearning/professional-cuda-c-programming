#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>

/*
 * An example usage of the cuFFT library. This example performs a 1D forward
 * FFT.
 */

int nprints = 30;

/*
 * Create N fake samplings along the function cos(x). These samplings will be
 * stored as single-precision floating-point values.
 */
void generate_fake_samples(int N, float **out)
{
    int i;
    float *result = (float *)malloc(sizeof(float) * N);
    double delta = M_PI / 20.0;

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

int main(int argc, char **argv)
{
    int i;
    int N = 2048;
    float *samples;
    cufftHandle plan = 0;
    cufftComplex *dComplexSamples, *complexSamples, *complexFreq;

    // Input Generation
    generate_fake_samples(N, &samples);
    real_to_complex(samples, &complexSamples, N);
    complexFreq = (cufftComplex *)malloc(
                      sizeof(cufftComplex) * N);
    printf("Initial Samples:\n");

    for (i = 0; i < nprints; i++)
    {
        printf("  %2.4f\n", samples[i]);
    }

    printf("  ...\n");

    // Setup the cuFFT plan
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    // Allocate device memory
    CHECK(cudaMalloc((void **)&dComplexSamples,
            sizeof(cufftComplex) * N));

    // Transfer inputs into device memory
    CHECK(cudaMemcpy(dComplexSamples, complexSamples,
            sizeof(cufftComplex) * N, cudaMemcpyHostToDevice));

    // Execute a complex-to-complex 1D FFT
    CHECK_CUFFT(cufftExecC2C(plan, dComplexSamples, dComplexSamples,
                             CUFFT_FORWARD));

    // Retrieve the results into host memory
    CHECK(cudaMemcpy(complexFreq, dComplexSamples,
            sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost));

    printf("Fourier Coefficients:\n");

    for (i = 0; i < nprints; i++)
    {
        printf("  %d: (%2.4f, %2.4f)\n", i + 1, complexFreq[i].x,
               complexFreq[i].y);
    }

    printf("  ...\n");

    free(samples);
    free(complexSamples);
    free(complexFreq);

    CHECK(cudaFree(dComplexSamples));
    CHECK_CUFFT(cufftDestroy(plan));

    return 0;
}
