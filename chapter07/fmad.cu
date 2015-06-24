#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * This example illustrates the effect on numerical accuracy of fusing a
 * multiply-add into a single MAD instruction.
 **/

__global__ void fmad_kernel(double x, double y, double *out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        *out = x * x + y;
    }
}

double host_fmad_kernel(double x, double y)
{
    return x * x + y;
}

int main(int argc, char **argv)
{
    double *d_out, h_out;
    double x = 2.891903;
    double y = -3.980364;

    double host_value = host_fmad_kernel(x, y);
    CHECK(cudaMalloc((void **)&d_out, sizeof(double)));
    fmad_kernel<<<1, 32>>>(x, y, d_out);
    CHECK(cudaMemcpy(&h_out, d_out, sizeof(double),
                     cudaMemcpyDeviceToHost));

    if (host_value == h_out)
    {
        printf("The device output the same value as the host.\n");
    }
    else
    {
        printf("The device output a different value than the host, diff=%e.\n",
               fabs(host_value - h_out));
    }

    return 0;
}
