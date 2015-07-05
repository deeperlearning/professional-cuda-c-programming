#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleavedFloat (float *g_idata, float *g_odata,
        unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    float *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8 (int *g_idata, int *g_odata,
        unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8Float (float *g_idata, float *g_odata,
        unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    float *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        float a1 = g_idata[idx];
        float a2 = g_idata[idx + blockDim.x];
        float a3 = g_idata[idx + 2 * blockDim.x];
        float a4 = g_idata[idx + 3 * blockDim.x];
        float b1 = g_idata[idx + 4 * blockDim.x];
        float b2 = g_idata[idx + 5 * blockDim.x];
        float b3 = g_idata[idx + 6 * blockDim.x];
        float b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile float *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 512;   // initial block size

    if(argc > 1)
    {
        blocksize = atoi(argv[1]);   // block size from command line argument
    }

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t fbytes = size * sizeof(float);
    float *h_f_idata = (float *) malloc(fbytes);
    float *h_f_odata = (float *) malloc(grid.x * sizeof(float));

    size_t ibytes = size * sizeof(int);
    int *h_i_idata = (int *) malloc(ibytes);
    int *h_i_odata = (int *) malloc(grid.x * sizeof(int));

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_f_idata[i] = (float)(rand() & 0xFF);
        h_i_idata[i] = (int)(rand() & 0xFF);
    }

    double iStart, iElaps;
    float f_gpu_sum = 0.0f;
    int i_gpu_sum = 0;

    // allocate device memory
    float *d_f_idata = NULL;
    float *d_f_odata = NULL;
    CHECK(cudaMalloc((void **) &d_f_idata, fbytes));
    CHECK(cudaMalloc((void **) &d_f_odata, grid.x * sizeof(float)));

    int *d_i_idata = NULL;
    int *d_i_odata = NULL;
    CHECK(cudaMalloc((void **) &d_i_idata, ibytes));
    CHECK(cudaMalloc((void **) &d_i_odata, grid.x * sizeof(int)));

    // reduceInterleaved
    CHECK(cudaMemcpy(d_i_idata, h_i_idata, ibytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_i_idata, d_i_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_i_odata, d_i_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    i_gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) i_gpu_sum += h_i_odata[i];

    printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, i_gpu_sum, grid.x, block.x);

    // reduceInterleavedFloat
    CHECK(cudaMemcpy(d_f_idata, h_f_idata, fbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceInterleavedFloat<<<grid, block>>>(d_f_idata, d_f_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_f_odata, d_f_odata, grid.x * sizeof(float),
                     cudaMemcpyDeviceToHost));
    f_gpu_sum = 0.0f;

    for (int i = 0; i < grid.x; i++) f_gpu_sum += h_f_odata[i];

    printf("gpu InterleavedFloat elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, f_gpu_sum, grid.x, block.x);

    // reduceCompleteUnrollWarsp8
    CHECK(cudaMemcpy(d_i_idata, h_i_idata, ibytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_i_idata, d_i_odata,
            size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_i_odata, d_i_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    i_gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) i_gpu_sum += h_i_odata[i];

    printf("gpu Cmptnroll8  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, i_gpu_sum, grid.x / 8, block.x);

    // reduceCompleteUnrollWarsp8Float
    CHECK(cudaMemcpy(d_f_idata, h_f_idata, fbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceCompleteUnrollWarps8Float<<<grid.x / 8, block>>>(d_f_idata, d_f_odata,
            size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_f_odata, d_f_odata, grid.x / 8 * sizeof(float),
                     cudaMemcpyDeviceToHost));
    f_gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) f_gpu_sum += h_f_odata[i];

    printf("gpu Cmptnroll8Float  elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, f_gpu_sum, grid.x / 8, block.x);


    // free host memory
    free(h_i_idata);
    free(h_i_odata);
    free(h_f_idata);
    free(h_f_odata);

    // free device memory
    CHECK(cudaFree(d_i_idata));
    CHECK(cudaFree(d_i_odata));
    CHECK(cudaFree(d_f_idata));
    CHECK(cudaFree(d_f_odata));

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
