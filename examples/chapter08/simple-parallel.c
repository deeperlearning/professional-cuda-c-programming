#include <stdio.h>
#include <stdlib.h>

/*
 * This example offers a brief introduction to the parallel directive. The
 * parallel directive executes a fixed number of threads throughout the code
 * block that follows it.  The programmer is responsible for using that
 * parallelism.
 */

#define N   1024

int main(int argc, char **argv)
{
    int i;
    /*
     * Note that this example does not require the restrict keyword that
     * simple-kernels.cu did. Because the parallel directive relies on the
     * programmer to mark parallelism, the compiler does not need to be careful
     * about multiple pointers referencing the same memory locations.
     */
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
     * Execute the following block of code on an accelerator, parallelizing the
     * two loops marked.
     */
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

    // Display part of the results
    for (i = 0; i < 10; i++)
    {
        printf("%d ", D[i]);
    }

    printf("...\n");

    return 0;
}
