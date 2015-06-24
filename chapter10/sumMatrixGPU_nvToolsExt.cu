#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>

#define WHITE   0XFFFFFF
#define SILVER  0XC0C0C0
#define GRAY    0X808080
#define BLACK   0X000000
#define YELLOW  0XFFFF00
#define FUCHSIA 0XFF00FF
#define RED     0XFF0000
#define MAROON  0X800000
#define LIME    0X00FF00
#define OLIVE   0X808000
#define GREEN   0X008000
#define PURPLE  0X800080
#define AQUA    0X00FFFF
#define TEAL    0X008080
#define BLUE    0X0000FF
#define NAVY    0X000080

void initialData(float *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];

        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (!match)
        printf("Arrays do not match.\n\n");
}

// grid 2D block 2D
__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx,
                             int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char **argv)
{
    printf("%s Starting ", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx, ny;
    int ishift = 12;

    if  (argc > 1) ishift = atoi(argv[1]);

    nx = ny = 1 << ishift;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);


    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;


    eventAttrib.color = RED;
    eventAttrib.message.ascii = "HostMalloc";
    nvtxRangeId_t hostMalloc = nvtxRangeStartEx(&eventAttrib);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    nvtxRangeEnd(hostMalloc);

    eventAttrib.color = YELLOW;
    eventAttrib.message.ascii = "SetData";
    nvtxRangeId_t setData = nvtxRangeStartEx(&eventAttrib);

    double iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = seconds() - iStart;
    printf("initialization: \t %f sec\n", iElaps);
    nvtxRangeEnd(setData);

    eventAttrib.color = PURPLE;
    eventAttrib.message.ascii = "ClearMemory";
    nvtxRangeId_t clearMemory = nvtxRangeStartEx(&eventAttrib);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    nvtxRangeEnd(clearMemory);

    // add matrix at host side for result checks
    eventAttrib.color = GREEN;
    eventAttrib.message.ascii = "HostSum";
    nvtxRangeId_t hostSum = nvtxRangeStartEx(&eventAttrib);

    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("sumMatrix on host:\t %f sec\n", iElaps);
    nvtxRangeEnd(hostSum);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    /*
     * init device data to 0.0f, then warm-up kernel to obtain accurate timing
     * result.
     */
    CHECK(cudaMemset(d_MatA, 0.0f, nBytes));
    CHECK(cudaMemset(d_MatB, 0.0f, nBytes));
    sumMatrixGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, 1, 1);


    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    iStart =  seconds();
    sumMatrixGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>> \n", iElaps,
           grid.x, grid.y, block.x, block.y);

    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    eventAttrib.color = NAVY;
    eventAttrib.message.ascii = "CompareResults";
    nvtxRangeId_t compResults = nvtxRangeStartEx(&eventAttrib);

    checkResult(hostRef, gpuRef, nxy);
    nvtxRangeEnd(compResults);

    eventAttrib.color = AQUA;
    eventAttrib.message.ascii = "ReleaseResource";
    nvtxRangeId_t releaseResource = nvtxRangeStartEx(&eventAttrib);
    // free device global memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    nvtxRangeEnd(releaseResource);

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}
