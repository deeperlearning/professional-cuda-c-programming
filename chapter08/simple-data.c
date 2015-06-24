#include <stdio.h>
#include <stdlib.h>

/*
 * This example offers a brief introduction to the data directive. The
 * data directive allows the programmer to explicitly mark variables to be
 * transferred to or from the accelerator. This serves as a performance
 * optimization by eliminating redundant or unnecessary memcpys.
 */

#define N   1024

int main(int argc, char **argv)
{
    int i;
    int *A = (int *)malloc(N * sizeof(int));
    int *B = (int *)malloc(N * sizeof(int));
    int *C = (int *)malloc(N * sizeof(int));
    int *D = (int *)malloc(N * sizeof(int));

    // Initialize A and B
    for (i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = 2 * i;
    }

    /*
     * Transfer the full contents of A and B to the accelerator, and transfer
     * the full contents of C and D back.
     */
#pragma acc data copyin(A[0:N], B[0:N]) copyout(C[0:N], D[0:N])
    {
#pragma acc parallel
        {
#pragma acc loop

            for (i = 0; i < N; i++)
            {
                C[i] = A[i] + B[i];
            }

#pragma acc loop

            for (i = 0; i < N; i++)
            {
                D[i] = C[i] * A[i];
            }
        }
    }

    // Display part of the results
    for (i = 0; i < 10; i++)
    {
        printf("%d ", D[i]);
    }

    printf("...\n");

    return 0;
}
