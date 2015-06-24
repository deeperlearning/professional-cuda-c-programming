#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * This example demonstrates the relative performance and accuracy of CUDA
 * standard and intrinsic functions.
 *
 * The computational kernel of this example is the iterative calculation of a
 * value squared. This computation is done on the host, on the device with a
 * standard function, and on the device with an intrinsic function. The results
 * from all three are compared for numerical accuracy (with the host as the
 * baseline), and the performance of standard and intrinsic functions is also
 * compared.
 **/

/**
 * Perform iters power operations using the standard powf function.
 **/
__global__ void standard_kernel(float a, float *out, int iters)
{
    int i;
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if(tid == 0)
    {
        float tmp;

        for (i = 0; i < iters; i++)
        {
            tmp = powf(a, 2.0f);
        }

        *out = tmp;
    }
}

/**
 * Perform iters power operations using the intrinsic __powf function.
 **/
__global__ void intrinsic_kernel(float a, float *out, int iters)
{
    int i;
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if(tid == 0)
    {
        float tmp;

        for (i = 0; i < iters; i++)
        {
            tmp = __powf(a, 2.0f);
        }

        *out = tmp;
    }
}

int main(int argc, char **argv)
{
    int i;
    int runs = 30;
    int iters = 1000;

    float *d_standard_out, h_standard_out;
    CHECK(cudaMalloc((void **)&d_standard_out, sizeof(float)));

    float *d_intrinsic_out, h_intrinsic_out;
    CHECK(cudaMalloc((void **)&d_intrinsic_out, sizeof(float)));

    float input_value = 8181.25;

    double mean_intrinsic_time = 0.0;
    double mean_standard_time = 0.0;

    for (i = 0; i < runs; i++)
    {
        double start_standard = seconds();
        standard_kernel<<<1, 32>>>(input_value, d_standard_out, iters);
        CHECK(cudaDeviceSynchronize());
        mean_standard_time += seconds() - start_standard;

        double start_intrinsic = seconds();
        intrinsic_kernel<<<1, 32>>>(input_value, d_intrinsic_out, iters);
        CHECK(cudaDeviceSynchronize());
        mean_intrinsic_time += seconds() - start_intrinsic;
    }

    CHECK(cudaMemcpy(&h_standard_out, d_standard_out, sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_intrinsic_out, d_intrinsic_out, sizeof(float),
                     cudaMemcpyDeviceToHost));
    float host_value = powf(input_value, 2.0f);

    printf("Host calculated\t\t\t%f\n", host_value);
    printf("Standard Device calculated\t%f\n", h_standard_out);
    printf("Intrinsic Device calculated\t%f\n", h_intrinsic_out);
    printf("Host equals Standard?\t\t%s diff=%e\n",
           host_value == h_standard_out ? "Yes" : "No",
           fabs(host_value - h_standard_out));
    printf("Host equals Intrinsic?\t\t%s diff=%e\n",
           host_value == h_intrinsic_out ? "Yes" : "No",
           fabs(host_value - h_intrinsic_out));
    printf("Standard equals Intrinsic?\t%s diff=%e\n",
           h_standard_out == h_intrinsic_out ? "Yes" : "No",
           fabs(h_standard_out - h_intrinsic_out));
    printf("\n");
    printf("Mean execution time for standard function powf:    %f s\n",
           mean_standard_time);
    printf("Mean execution time for intrinsic function __powf: %f s\n",
           mean_intrinsic_time);

    return 0;
}
