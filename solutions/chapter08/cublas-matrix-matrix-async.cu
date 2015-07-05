#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cublas_v2.h"

/*
 * A simple example of performing matrix-vector multiplication using the cuBLAS
 * library and some randomly generated inputs.
 */

/*
 * M = # of rows
 * N = # of columns
 */
int M = 1024;
int N = 1024;

/*
 * Generate a matrix with M rows and N columns in column-major order. The matrix
 * will be filled with random single-precision floating-point values between 0
 * and 100.
 */
void generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);

    // For each column
    for (j = 0; j < N; j++)
    {
        // For each row
        for (i = 0; i < M; i++)
        {
            double dr = (double)rand();
            A[j * M + i] = (dr / rMax) * 100.0;
        }
    }

    *outA = A;
}

int main(int argc, char **argv)
{
    int i, j;
    float *A, *dA;
    float *B, *dB;
    float *C, *dC;
    float beta;
    float alpha;
    cublasHandle_t handle = 0;
    cudaStream_t stream = 0;

    alpha = 3.0f;
    beta = 4.0f;

    // Generate inputs
    srand(9384);
    generate_random_dense_matrix(M, N, &A);
    generate_random_dense_matrix(N, M, &B);
    C = (float *)malloc(sizeof(float) * M * M);
    memset(C, 0x00, sizeof(float) * M * M);

    // Create the cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    // Allocate device memory
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&dB, sizeof(float) * N * M));
    CHECK(cudaMalloc((void **)&dC, sizeof(float) * M * M));

    // Transfer inputs to the device
    CHECK_CUBLAS(cublasSetMatrixAsync(M, N, sizeof(float), A, M, dA, M,
                stream));
    CHECK_CUBLAS(cublasSetMatrixAsync(N, M, sizeof(float), B, N, dB, N,
                stream));
    CHECK_CUBLAS(cublasSetMatrixAsync(M, M, sizeof(float), C, M, dC, M,
                stream));

    // Execute the matrix-vector multiplication
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, M, N, &alpha,
                dA, M, dB, N, &beta, dC, M));

    // Retrieve the output vector from the device
    CHECK_CUBLAS(cublasGetMatrixAsync(M, M, sizeof(float), dC, M, C, M,
                stream));
    CHECK(cudaStreamSynchronize(stream));

    for (j = 0; j < 10; j++)
    {
        for (i = 0; i < 10; i++)
        {
            printf("%2.2f ", C[j * M + i]);
        }
        printf("...\n");
    }

    printf("...\n");

    free(A);
    free(B);
    free(C);

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    CHECK(cudaStreamDestroy(stream));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}
