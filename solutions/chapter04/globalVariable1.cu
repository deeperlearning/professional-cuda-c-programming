#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * An example of using a statically declared global variable (devData) to store
 * a floating-point value on the device.
 */

__device__ float devData[5];

__global__ void checkGlobalVariable()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // display the original value
    printf("Device: the value of the global variable is %f\n", devData[tid]);

    // alter the value
    if (tid < 5)
    {
        devData[tid] *= tid;
    }
}

int main(void)
{
    // initialize the global variable
    float values[5] = { 3.14f, 3.14f, 3.14f, 3.14f, 3.14f };
    CHECK(cudaMemcpyToSymbol(devData, values, 5 * sizeof(float)));
    printf("Host:   copied [ %f %f %f %f %f ] to the global variable\n",
            values[0], values[1], values[2], values[3], values[4]);

    // invoke the kernel
    checkGlobalVariable<<<1, 5>>>();

    // copy the global variable back to the host
    CHECK(cudaMemcpyFromSymbol(values, devData, 5 * sizeof(float)));
    printf("Host:   the values changed by the kernel to [ %f %f %f %f %f ]\n",
            values[0], values[1], values[2], values[3], values[4]);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
