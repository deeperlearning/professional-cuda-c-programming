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
int M = 1 << 15;
int N = 1 << 15;

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
    if (A == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

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

void generate_random_dense_matrix_iter(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);
    if (A == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // For each column
    for (i = 0; i < M; i++)
    {
        // For each row
        for (j = 0; j < N; j++)
        {
            double dr = (double)rand();
            A[j * M + i] = (dr / rMax) * 100.0;
        }
    }

    *outA = A;
}

void generate_random_dense_matrix_row(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);
    if (A == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // For each column
    for (i = 0; i < M; i++)
    {
        // For each row
        for (j = 0; j < N; j++)
        {
            double dr = (double)rand();
            A[i * M + j] = (dr / rMax) * 100.0;
        }
    }

    *outA = A;
}

int main(int argc, char **argv)
{
    float *A;

    srand(9384);

    double start = seconds();
    generate_random_dense_matrix(M, N, &A);
    double elapsed = seconds() - start;
    printf("Original execution time %.10f seconds\n", elapsed);

    start = seconds();
    generate_random_dense_matrix_iter(M, N, &A);
    elapsed = seconds() - start;
    printf("Re-ordered loops execution time %.10f seconds\n", elapsed);

    start = seconds();
    generate_random_dense_matrix_row(M, N, &A);
    elapsed = seconds() - start;
    printf("Re-ordered loops execution time %.10f seconds\n", elapsed);

    return 0;
}
