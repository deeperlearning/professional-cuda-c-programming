#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * This example demonstrates floating-point's inability to represent certain
 * values with a specific value as an example.
 *
 * In this example, the value 12.1 is stored in single- and double-precision
 * floating-point variables on both the host and device. After retrieving the
 * results from the device, the actual values stored are printed to 20 decimal
 * places and the single- and double-precision results from the host and device
 * are compared to each other to verify that host and device are equally
 * accurate for the same type.
 **/

/**
 * Save the single- and double-precision representation of 12.1 from the device
 * into global memory. That global memory is then copied back to the host for
 * later analysis.
 **/
__global__ void kernel(float *F, double *D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        *F = 12.1;
        *D = 12.1;
    }
}

int main(int argc, char **argv)
{
    float *deviceF;
    float h_deviceF;
    double *deviceD;
    double h_deviceD;

    float hostF = 12.1;
    double hostD = 12.1;

    CHECK(cudaMalloc((void **)&deviceF, sizeof(float)));
    CHECK(cudaMalloc((void **)&deviceD, sizeof(double)));
    kernel<<<1, 32>>>(deviceF, deviceD);
    CHECK(cudaMemcpy(&h_deviceF, deviceF, sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_deviceD, deviceD, sizeof(double),
                     cudaMemcpyDeviceToHost));

    printf("Host single-precision representation of 12.1   = %.20f\n", hostF);
    printf("Host double-precision representation of 12.1   = %.20f\n", hostD);
    printf("Device single-precision representation of 12.1 = %.20f\n", hostF);
    printf("Device double-precision representation of 12.1 = %.20f\n", hostD);
    printf("Device and host single-precision representation equal? %s\n",
           hostF == h_deviceF ? "yes" : "no");
    printf("Device and host double-precision representation equal? %s\n",
           hostD == h_deviceD ? "yes" : "no");

    return 0;
}

