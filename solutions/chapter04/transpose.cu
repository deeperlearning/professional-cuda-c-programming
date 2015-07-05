#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Various memory access pattern optimizations applied to a matrix transpose
 * kernel.
 */

#define BDIMX 16
#define BDIMY 16

void initialData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)( rand() & 0xFF ) / 10.0f; //100.0f;
    }

    return;
}

void printData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%dth element: %f\n", i, in[i]);
    }

    return;
}

void checkResult(float *hostRef, float *gpuRef, const int size, int showme)
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

        if (showme && i > size / 2 && i < size / 2 + 5)
        {
            // printf("%dth element: host %f gpu %f\n",i,hostRef[i],gpuRef[i]);
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

__global__ void warmup(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

// case 0 copy kernel: access data in rows
__global__ void copyRow(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

// case 1 copy kernel: access data in columns
__global__ void copyCol(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[ix * ny + iy];
    }
}

// case 2 transpose kernel: read in rows and write in columns
__global__ void transposeNaiveRow(float *out, float *in, const int nx,
                                  const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

// case 3 transpose kernel: read in columns and write in rows
__global__ void transposeNaiveCol(float *out, float *in, const int nx,
                                  const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

// case 4 transpose kernel: read in rows and write in columns + unroll 4 blocks
__global__ void transposeUnroll4Row(float *out, float *in, const int nx,
                                    const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny)
    {
        out[to]                   = in[ti];
        out[to + ny * blockDim.x]   = in[ti + blockDim.x];
        out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
        out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
    }
}

__global__ void transposeRow(float *out, float *in, const int nx,
                                    const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int row = iy * gridDim.x * blockDim.x + ix;

    if (row < ny)
    {
        int row_start = row * nx;
        int row_end = (row + 1) * nx;
        int col_index = row;
        for (int i = row_start; i < row_end; i++) {
            out[col_index] = in[i];
            col_index += nx;
        }
    }
}

__global__ void transposeUnroll8Row(float *out, float *in, const int nx,
                                    const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 7 * blockDim.x < nx && iy < ny)
    {
        out[to]                   = in[ti];
        out[to + ny * blockDim.x]   = in[ti + blockDim.x];
        out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
        out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
        out[to + ny * 4 * blockDim.x] = in[ti + 4 * blockDim.x];
        out[to + ny * 5 * blockDim.x] = in[ti + 5 * blockDim.x];
        out[to + ny * 6 * blockDim.x] = in[ti + 6 * blockDim.x];
        out[to + ny * 7 * blockDim.x] = in[ti + 7 * blockDim.x];
    }
}

// case 5 transpose kernel: read in columns and write in rows + unroll 4 blocks
__global__ void transposeUnroll4Col(float *out, float *in, const int nx,
                                    const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny)
    {
        out[ti]                = in[to];
        out[ti +   blockDim.x] = in[to +   blockDim.x * ny];
        out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
        out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
    }
}

/*
 * case 6 :  transpose kernel: read in rows and write in colunms + diagonal
 * coordinate transform
 */
__global__ void transposeDiagonalRow(float *out, float *in, const int nx,
                                     const int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

/*
 * case 7 :  transpose kernel: read in columns and write in row + diagonal
 * coordinate transform.
 */
__global__ void transposeDiagonalCol(float *out, float *in, const int nx,
                                     const int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

__global__ void transposeDiagonalColUnroll4(float *out, float *in, const int nx,
                                     const int ny)
{
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix_stride = blockDim.x * blk_x;
    unsigned int ix = ix_stride * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
        out[iy * nx + ix + blockDim.x] = in[(ix + blockDim.x) * ny + iy];
        out[iy * nx + ix + 2 * blockDim.x] =
            in[(ix + 2 * blockDim.x) * ny + iy];
        out[iy * nx + ix + 3 * blockDim.x] =
            in[(ix + 3 * blockDim.x) * ny + iy];
    }
}

// main functions
int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up array size 2048
    int nx = 1 << 11;
    int ny = 1 << 11;

    // select a kernel and block size
    int iKernel = 0;
    int blockx = 16;
    int blocky = 16;

    if (argc > 1) iKernel = atoi(argv[1]);

    if (argc > 2) blockx  = atoi(argv[2]);

    if (argc > 3) blocky  = atoi(argv[3]);

    if (argc > 4) nx  = atoi(argv[4]);

    if (argc > 5) ny  = atoi(argv[5]);

    printf(" with matrix nx %d ny %d with kernel %d\n", nx, ny, iKernel);
    size_t nBytes = nx * ny * sizeof(float);

    // execution configuration
    dim3 block (blockx, blocky);
    dim3 grid  ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    dim3 grid4 ((nx + block.x - 1) / (block.x * 4), (ny + block.y - 1) /
            (block.y * 4));
    dim3 grid8 ((nx + block.x - 1) / (block.x * 8), (ny + block.y - 1) /
            (block.y * 8));

    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    // initialize host array
    initialData(h_A, nx * ny);

    // transpose at host side
    transposeHost(hostRef, h_A, nx, ny);

    // allocate device memory
    float *d_A, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // warmup to avoide startup overhead
    double iStart = seconds();
    warmup<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup         elapsed %f sec\n", iElaps);
    CHECK(cudaGetLastError());

    // transposeNaiveRow
    iStart = seconds();
    transposeNaiveRow<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    printf("transposeNaiveRow elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>>\n", iElaps,
           grid.x, grid.y, block.x, block.y);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nx * ny, 1);

    // transposeNaiveCol
    iStart = seconds();
    transposeNaiveCol<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    printf("transposeNaiveCol elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>>\n", iElaps,
           grid.x, grid.y, block.x, block.y);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nx * ny, 1);

    // transposeUnroll4Row
    iStart = seconds();
    transposeUnroll4Row<<<grid4, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    printf("transposeUnroll4Row elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>>\n", iElaps,
           grid.x, grid.y, block.x, block.y);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nx * ny, 1);

    // transposeRow
    iStart = seconds();
    transposeRow<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    printf("transposeRow elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>>\n", iElaps,
           grid.x, grid.y, block.x, block.y);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nx * ny, 1);

    // transposeUnroll8Row
    iStart = seconds();
    transposeUnroll8Row<<<grid8, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    printf("transposeUnroll8Row elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>>\n", iElaps,
           grid.x, grid.y, block.x, block.y);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nx * ny, 1);

    // transposeUnroll4Col
    iStart = seconds();
    transposeUnroll4Col<<<grid4, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    printf("transposeUnroll4Col elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>>\n", iElaps,
           grid.x, grid.y, block.x, block.y);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nx * ny, 1);

    // transposeDiagonalRow
    iStart = seconds();
    transposeDiagonalRow<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    printf("transposeDiagonalRow elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>>\n", iElaps,
           grid.x, grid.y, block.x, block.y);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nx * ny, 1);

    // transposeDiagonalCol
    iStart = seconds();
    transposeDiagonalCol<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    printf("transposeDiagonalCol elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>>\n", iElaps,
           grid.x, grid.y, block.x, block.y);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nx * ny, 1);

    // transposeDiagonalColUnroll4
    iStart = seconds();
    transposeDiagonalColUnroll4<<<grid4, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    printf("transposeDiagonalColUnroll4 elapsed %f sec <<< grid (%d,%d) block (%d,%d)>>>\n", iElaps,
           grid.x, grid.y, block.x, block.y);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nx * ny, 1);


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
