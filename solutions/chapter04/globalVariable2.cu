#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * An example of using a statically declared global variable (devData) to store
 * a floating-point value on the device.
 */

__device__ float devData[1];

__global__ void checkGlobalVariable()
{
    // display the original value
    printf("Device: the value of the global variable is %f\n", devData[0]);

    // alter the value
    devData[0] += 2.0f;
}

int main(void)
{
    // initialize the global variable
    float value = 3.14f;
    float *d_ptr;
    CHECK(cudaGetSymbolAddress((void **)&d_ptr, devData));
    CHECK(cudaMemcpy(d_ptr, &value, sizeof(float), cudaMemcpyHostToDevice));
    printf("Host:   copied %f to the global variable\n", value);

    // invoke the kernel
    checkGlobalVariable<<<1, 1>>>();

    // copy the global variable back to the host
    CHECK(cudaMemcpy(&value, d_ptr, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Host:   the value changed by the kernel to %f\n", value);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
