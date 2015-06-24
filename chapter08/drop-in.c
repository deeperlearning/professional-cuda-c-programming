#include <stdio.h>
#include <stdlib.h>

extern int sgemm_(char *transa, char *transb, int *m, int *
                  n, int *k, float *alpha, float *a, int *lda, float *b, int *
                  ldb, float *beta, float *c, int *ldc);

/*
 * A simple example of re-compiling legacy BLAS code to use the drop-in cuBLAS
 * library.
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
    float *A, *B, *C;
    float alpha = 3.0f;
    float beta = 4.0f;

    // Generate inputs
    srand(9384);
    generate_random_dense_matrix(M, N, &A);
    generate_random_dense_matrix(N, M, &B);
    generate_random_dense_matrix(M, N, &C);

    sgemm_("N", "N", &M, &M, &N, &alpha, A, &M, B, &N, &beta, C, &M);

    for (i = 0; i < 10; i++)
    {
        for (j = 0; j < 10; j++)
        {
            printf("%2.2f ", C[j * M + i]);
        }

        printf("...\n");
    }

    printf("...\n");

    free(A);
    free(B);
    free(C);

    return 0;
}
