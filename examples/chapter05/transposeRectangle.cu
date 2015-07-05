#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Example kernels for transposing a rectangular host array using a variety of
 * optimizations, including shared memory, unrolling, and memory padding.
 */

#define BDIMX 16
#define BDIMY 16

#define IPAD 2

void initialData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void printData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%3.0f ", in[i]);
    }

    printf("\n");
    return;
}

void checkResult(float *hostRef, float *gpuRef, const int size)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < size; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                   gpuRef[i]);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

void transposeHost(float *out, float *in, const int nx, const int ny)
{
    for( int iy = 0; iy < ny; ++iy)
    {
        for( int ix = 0; ix < nx; ++ix)
        {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

__global__ void copyGmem(float *out, float *in, const int nx, const int ny)
{
    // matrix coordinate (ix,iy)
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // transpose with boundary test
    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

__global__ void naiveGmem(float *out, float *in, const int nx, const int ny)
{
    // matrix coordinate (ix,iy)
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // transpose with boundary test
    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__ void naiveGmemUnroll(float *out, float *in, const int nx,
                                const int ny)
{
    unsigned int ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int ti = iy * nx + ix;
    unsigned int to = ix * ny + iy;

    if (ix + blockDim.x < nx && iy < ny)
    {
        out[to]                   = in[ti];
        out[to + ny * blockDim.x]   = in[ti + blockDim.x];
    }
}


__global__ void transposeSmemUnroll(float *out, float *in, const int nx,
                                    const int ny)
{
    // static 1D shared memory
    __shared__ float tile[BDIMY * BDIMX * 2];

    // coordinate in original matrix
    unsigned int ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // linear global memory index for original matrix
    unsigned int ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    unsigned int ix2 = blockIdx.y * blockDim.y + icol;
    unsigned int iy2 = 2 * blockIdx.x * blockDim.x + irow;

    // linear global memory index for transposed matrix
    unsigned int to = iy2 * ny + ix2;

    if (ix + blockDim.x < nx && iy < ny)
    {
        // load two rows from global memory to shared memory
        unsigned int row_idx = 2 * threadIdx.y * blockDim.x + threadIdx.x;
        tile[row_idx]       = in[ti];
        tile[row_idx + BDIMX] = in[ti + BDIMX];

        // thread synchronization
        __syncthreads();

        // store two rows to global memory from two columns of shared memory
        unsigned int col_idx = icol * blockDim.x * 2 + irow;
        out[to] = tile[col_idx];
        out[to + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}

__global__ void transposeSmemUnrollPad(float *out, float *in, const int nx,
                                       const int ny)
{
    // static 1D shared memory with padding
    __shared__ float tile[BDIMY * (BDIMX * 2 + IPAD)];

    // coordinate in original matrix
    unsigned int ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // linear global memory index for original matrix
    unsigned int ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    unsigned int ix2 = blockIdx.y * blockDim.y + icol;
    unsigned int iy2 = 2 * blockIdx.x * blockDim.x + irow;

    // linear global memory index for transposed matrix
    unsigned int to = iy2 * ny + ix2;

    if (ix + blockDim.x < nx && iy < ny)
    {
        // load two rows from global memory to shared memory
        unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) +
            threadIdx.x;
        tile[row_idx]         = in[ti];
        tile[row_idx + BDIMX] = in[ti + BDIMX];

        // thread synchronization
        __syncthreads();

        // store two rows to global memory from two columns of shared memory
        unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;
        out[to] = tile[col_idx];
        out[to + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}

__global__ void transposeSmemUnrollPadDyn (float *out, float *in, const int nx,
        const int ny)
{
    // dynamic shared memory
    extern __shared__ float tile[];

    unsigned int ix = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix;

    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    unsigned int ix2 = blockDim.y * blockIdx.y + icol;
    unsigned int iy2 = blockDim.x * 2 * blockIdx.x + irow;
    unsigned int to = iy2 * ny + ix2;

    // transpose with boundary test
    if (ix + blockDim.x < nx && iy < ny)
    {
        // load data from global memory to shared memory
        unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) +
            threadIdx.x;
        tile[row_idx]       = in[ti];
        tile[row_idx + BDIMX] = in[ti + BDIMX];

        // thread synchronization
        __syncthreads();

        unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;
        out[to] = tile[col_idx];
        out[to + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}

__global__ void transposeSmem(float *out, float *in, int nx, int ny)
{
    // static shared memory
    __shared__ float tile[BDIMY][BDIMX];

    // coordinate in original matrix
    unsigned int ix, iy, ti, to;
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // linear global memory index for original matrix
    ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    ix = blockDim.y * blockIdx.y + icol;
    iy = blockDim.x * blockIdx.x + irow;

    // linear global memory index for transposed matrix
    to = iy * ny + ix;

    // transpose with boundary test
    if (ix < nx && iy < ny)
    {
        // load data from global memory to shared memory
        tile[threadIdx.y][threadIdx.x] = in[ti];

        // thread synchronization
        __syncthreads();

        // store data to global memory from shared memory
        out[to] = tile[icol][irow];
    }
}

__global__ void transposeSmemPad(float *out, float *in, int nx, int ny)
{
    // static shared memory with padding
    __shared__ float tile[BDIMY][BDIMX + IPAD];

    // coordinate in original matrix
    unsigned int  ix, iy, ti, to;
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // linear global memory index for original matrix
    ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    ix = blockDim.y * blockIdx.y + icol;
    iy = blockDim.x * blockIdx.x + irow;

    // linear global memory index for transposed matrix
    to = iy * ny + ix;

    // transpose with boundary test
    if (ix < nx && iy < ny)
    {
        // load data from global memory to shared memory
        tile[threadIdx.y][threadIdx.x] = in[ti];

        // thread synchronization
        __syncthreads();

        // store data to global memory from shared memory
        out[to] = tile[icol][irow];
    }
}

__global__ void transposeSmemDyn(float *out, float *in, int nx, int ny)
{
    // dynamic shared memory
    extern __shared__ float tile[];

    // coordinate in original matrix
    unsigned int  ix, iy, ti, to;
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // linear global memory index for original matrix
    ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int row_idx, col_idx, irow, icol;
    row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    irow    = row_idx / blockDim.y;
    icol    = row_idx % blockDim.y;
    col_idx = icol * blockDim.x + irow;

    // coordinate in transposed matrix
    ix = blockDim.y * blockIdx.y + icol;
    iy = blockDim.x * blockIdx.x + irow;

    // linear global memory index for transposed matrix
    to = iy * ny + ix;

    // transpose with boundary test
    if (ix < nx && iy < ny)
    {
        // load data from global memory to shared memory
        tile[row_idx] = in[ti];

        // thread synchronization
        __syncthreads();

        // store data to global memory from shared memory
        out[to] = tile[col_idx];
    }
}

__global__ void transposeSmemPadDyn(float *out, float *in, int nx, int ny)
{
    // static shared memory with padding
    extern __shared__ float tile[];

    // coordinate in original matrix
    unsigned int  ix, iy, ti, to;
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // linear global memory index for original matrix
    ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int irow    = idx / blockDim.y;
    unsigned int icol    = idx % blockDim.y;
    unsigned int col_idx = icol * (blockDim.x + IPAD) + irow;

    // coordinate in transposed matrix
    ix = blockDim.y * blockIdx.y + icol;
    iy = blockDim.x * blockIdx.x + irow;

    // linear global memory index for transposed matrix
    to = iy * ny + ix;

    // transpose with boundary test
    if (ix < nx && iy < ny)
    {
        // load data from global memory to shared memory
        tile[row_idx] = in[ti];

        // thread synchronization
        __syncthreads();

        // store data to global memory from shared memory
        out[to] = tile[col_idx];
    }
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool iprint = 0;

    // set up array size 2048
    int nx = 1 << 12;
    int ny = 1 << 12;

    if (argc > 1) iprint = atoi(argv[1]);

    if (argc > 2) nx  = atoi(argv[2]);

    if (argc > 3) ny  = atoi(argv[3]);

    printf(" with matrix nx %d ny %d\n", nx, ny);
    size_t nBytes = nx * ny * sizeof(float);

    // execution configuration
    dim3 block (BDIMX, BDIMY);
    dim3 grid  ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    dim3 grid2  ((nx + block.x * 2 - 1) / (block.x * 2),
                 (ny + block.y - 1) / block.y);

    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    //  initialize host array
    initialData(h_A, nx * ny);

    //  transpose at host side
    transposeHost(hostRef, h_A, nx, ny);

    // allocate device memory
    float *d_A, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // tranpose gmem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    double iStart = seconds();
    copyGmem<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, nx * ny);

    float ibnd = 2 * nx * ny * sizeof(float) / (1024.0 * 1024.0 * 1024.0) /
        iElaps;
    ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("copyGmem elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n", iElaps, grid.x, grid.y, block.x,
           block.y, ibnd);

    // tranpose gmem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    naiveGmem<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, nx * ny);

    ibnd = 2 * nx * ny * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("naiveGmem elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n", iElaps, grid.x, grid.y, block.x,
           block.y, ibnd);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    naiveGmemUnroll<<<grid2, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, nx * ny);

    checkResult(hostRef, gpuRef, nx * ny);
    ibnd = 2 * nx * ny * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("naiveGmemUnroll elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n", iElaps, grid2.x, grid2.y, block.x,
           block.y, ibnd);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    transposeSmem<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, nx * ny);

    checkResult(hostRef, gpuRef, nx * ny);
    ibnd = 2 * nx * ny * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("transposeSmem elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n", iElaps, grid.x, grid.y, block.x,
           block.y, ibnd);

    // tranpose smem pad
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    transposeSmemPad<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, nx * ny);

    checkResult(hostRef, gpuRef, nx * ny );
    ibnd = 2 * nx * ny * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("transposeSmemPad elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n", iElaps, grid.x, grid.y, block.x,
           block.y, ibnd);

    // tranpose smem pad
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    transposeSmemDyn<<<grid, block, BDIMX*BDIMY*sizeof(float)>>>(d_C, d_A, nx,
            ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, nx * ny);

    checkResult(hostRef, gpuRef, nx * ny);
    ibnd = 2 * nx * ny * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("transposeSmemDyn elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n", iElaps, grid.x, grid.y, block.x,
           block.y, ibnd);

    // tranpose smem pad
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
  transposeSmemPadDyn<<<grid, block, (BDIMX + IPAD) * BDIMY * sizeof(float)>>>(
          d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, nx * ny);

    checkResult(hostRef, gpuRef, nx * ny );
    ibnd = 2 * nx * ny * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("transposeSmemPadDyn elapsed %f sec <<< grid (%d,%d) block "
           "(%d,%d)>>> effective bandwidth %f GB\n", iElaps, grid.x, grid.y,
           block.x, block.y, ibnd);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    transposeSmemUnroll<<<grid2, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, nx * ny);

    checkResult(hostRef, gpuRef, nx * ny );
    ibnd = 2 * nx * ny * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("transposeSmemUnroll elapsed %f sec <<< grid (%d,%d) block "
           "(%d,%d)>>> effective bandwidth %f GB\n", iElaps, grid2.x, grid2.y,
           block.x, block.y, ibnd);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    transposeSmemUnrollPad<<<grid2, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, nx * ny);

    checkResult(hostRef, gpuRef, nx * ny );
    ibnd = 2 * nx * ny * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("transposeSmemUnrollPad elapsed %f sec <<< grid (%d,%d) block "
           "(%d,%d)>>> effective bandwidth %f GB\n", iElaps, grid2.x, grid2.y,
           block.x, block.y, ibnd);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    iStart = seconds();
    transposeSmemUnrollPadDyn<<<grid2, block, (BDIMX * 2 + IPAD) * BDIMY *
        sizeof(float)>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, nx * ny);

    checkResult(hostRef, gpuRef, nx * ny );
    ibnd = 2 * nx * ny * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / iElaps;
    ibnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("transposeSmemUnrollPadDyn elapsed %f sec <<< grid (%d,%d) block "
           "(%d,%d)>>> effective bandwidth %f GB\n", iElaps, grid2.x, grid2.y,
           block.x, block.y, ibnd);

    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
