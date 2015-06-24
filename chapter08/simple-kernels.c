#include <stdio.h>
#include <stdlib.h>

/*
 * This example offers a brief introduction to the kernels directive. The
 * kernels directive attempts to break the code block that follows into
 * accelerator kernels, generally by searching for parallelizable loops. It then
 * launches each kernel on the acclerator using an automatically configured
 * thread configuration.
 */

#define N   1024

int main(int argc, char **argv)
{
    int i;
    /*
     * restrict indicates to the compiler that the memory pointed to by A, B, C,
     * and D will only be accessed through those respective pointers or by
     * offsets from those pointers. This restriction makes it possible to
     * analyze the loops below for parallelization.
     */
    int *restrict A = (int *)malloc(N * sizeof(int));
    int *restrict B = (int *)malloc(N * sizeof(int));
    int *restrict C = (int *)malloc(N * sizeof(int));
    int *restrict D = (int *)malloc(N * sizeof(int));

    // Initialize A and B
    for (i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = 2 * i;
    }

    // Execute the following block of code on an accelerator
#pragma acc kernels
    {
        for (i = 0; i < N; i++)
        {
            C[i] = A[i] + B[i];
        }

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
