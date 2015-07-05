#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cublas_v2.h>

/*
 * This example illustrates the use of OpenACC and CUDA libraries in the same
 * application. cuRAND is used to fill two input matrices with random values.
 * OpenACC is used to implement a matrix-multiply using the parallel and loop
 * directives. Finally, cuBLAS is used to first sum the values of every row, and
 * then sum those values together to calculate the sum of all values in the
 * output matrix.
 */

#define M   1024
#define N   1024
#define P   1024

int main(int argc, char **argv)
{
    int i, j, k;
    float *__restrict__ d_A;
    float *__restrict__ d_B;
    float *__restrict__ d_C;
    float *d_row_sums;
    float total_sum;
    curandGenerator_t rand_state = 0;
    cublasHandle_t cublas_handle = 0;

    // Initialize the cuRAND and cuBLAS handles.
    CHECK_CURAND(curandCreateGenerator(&rand_state, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    // Allocate GPU memory for the input matrices, output matrix, and row sums.
    CHECK(cudaMalloc((void **)&d_A, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&d_B, sizeof(float) * N * P));
    CHECK(cudaMalloc((void **)&d_C, sizeof(float) * M * P));
    CHECK(cudaMalloc((void **)&d_row_sums, sizeof(float) * M));

    // Generate random values in both input matrices.
    CHECK_CURAND(curandGenerateUniform(rand_state, d_A, M * N));
    CHECK_CURAND(curandGenerateUniform(rand_state, d_B, N * P));

    // Perform a matrix multiply parallelized across gangs and workers
#pragma acc parallel loop gang deviceptr(d_A, d_B, d_C)

    for (i = 0; i < M; i++)
    {
#pragma acc loop worker vector

        for (j = 0; j < P; j++)
        {
            float sum = 0.0f;

            for (k = 0; k < N; k++)
            {
                sum += d_A[i * N + k] * d_B[k * P + j];
            }

            d_C[i * P + j] = sum;
        }
    }

    /*
     * Set cuBLAS to device pointer mode, indicating that all scalars are passed
     * as device pointers.
     */
    CHECK_CUBLAS(cublasSetPointerMode(cublas_handle,
                                      CUBLAS_POINTER_MODE_DEVICE));

    // Sum the values contained in each row.
    for (i = 0; i < M; i++)
    {
        CHECK_CUBLAS(cublasSasum(cublas_handle, P, d_C + (i * P), 1,
                                 d_row_sums + i));
    }

    /*
     * Set cuBLAS back to host pointer mode, indicating that all scalars are
     * passed as host pointers.
     */
    CHECK_CUBLAS(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
    /*
     * Do the final sum of the sum of all rows to produce a total for the whole
     * output matrix.
     */
    CHECK_CUBLAS(cublasSasum(cublas_handle, M, d_row_sums, 1, &total_sum));
    CHECK(cudaDeviceSynchronize());

    // Release device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_row_sums));

    printf("Total sum = %f\n", total_sum);

    return 0;
}
