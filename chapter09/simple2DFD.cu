#include "../common/common.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
 * This example implements a 2D stencil computation, spreading the computation
 * across multiple GPUs. This requires communicating halo regions between GPUs
 * on every iteration of the stencil as well as managing multiple GPUs from a
 * single host application. Here, kernels and transfers are issued in
 * breadth-first order to each CUDA stream. Each CUDA stream is associated with
 * a single CUDA device.
 */

#define a0     -3.0124472f
#define a1      1.7383092f
#define a2     -0.2796695f
#define a3      0.0547837f
#define a4     -0.0073118f

// cnst for gpu
#define BDIMX       32
#define NPAD        4
#define NPAD2       8

// constant memories for 8 order FD coefficients
__device__ __constant__ float coef[5];

// set up fd coefficients
void setup_coef (void)
{
    const float h_coef[] = {a0, a1, a2, a3, a4};
    CHECK( cudaMemcpyToSymbol( coef, h_coef, 5 * sizeof(float) ));
}

void saveSnapshotIstep(
    int istep,
    int nx,
    int ny,
    int ngpus,
    float **g_u2)
{
    float *iwave = (float *)malloc(nx * ny * sizeof(float));

    if (ngpus > 1)
    {
        unsigned int skiptop = nx * 4;
        unsigned int gsize = nx * ny / 2;

        for (int i = 0; i < ngpus; i++)
        {
            CHECK(cudaSetDevice(i));
            int iskip = (i == 0 ? 0 : skiptop);
            int ioff  = (i == 0 ? 0 : gsize);
            CHECK(cudaMemcpy(iwave + ioff, g_u2[i] + iskip,
                        gsize * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    else
    {
        unsigned int isize = nx * ny;
        CHECK(cudaMemcpy (iwave, g_u2[0], isize * sizeof(float),
                          cudaMemcpyDeviceToHost));
    }

    char fname[20];
    sprintf(fname, "snap_at_step_%d", istep);

    FILE *fp_snap = fopen(fname, "w");

    fwrite(iwave, sizeof(float), nx * ny, fp_snap);
    printf("%s: nx = %d ny = %d istep = %d\n", fname, nx, ny, istep);
    fflush(stdout);
    fclose(fp_snap);

    free(iwave);
    return;
}

inline bool isCapableP2P(int ngpus)
{
    cudaDeviceProp prop[ngpus];
    int iCount = 0;

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaGetDeviceProperties(&prop[i], i));

        if (prop[i].major >= 2) iCount++;

        printf("> GPU%d: %s %s capable of Peer-to-Peer access\n", i,
                prop[i].name, (prop[i].major >= 2 ? "is" : "not"));
        fflush(stdout);
    }

    if(iCount != ngpus)
    {
        printf("> no enough device to run this application\n");
        fflush(stdout);
    }

    return (iCount == ngpus);
}

/*
 * enable P2P memcopies between GPUs (all GPUs must be compute capability 2.0 or
 * later (Fermi or later))
 */
inline void enableP2P (int ngpus)
{
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        for (int j = 0; j < ngpus; j++)
        {
            if (i == j) continue;

            int peer_access_available = 0;
            CHECK(cudaDeviceCanAccessPeer(&peer_access_available, i, j));

            if (peer_access_available) CHECK(cudaDeviceEnablePeerAccess(j, 0));
        }
    }
}

inline bool isUnifiedAddressing (int ngpus)
{
    cudaDeviceProp prop[ngpus];

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaGetDeviceProperties(&prop[i], i));
    }

    const bool iuva = (prop[0].unifiedAddressing && prop[1].unifiedAddressing);
    printf("> GPU%d: %s %s unified addressing\n", 0, prop[0].name,
           (prop[0].unifiedAddressing ? "support" : "not support"));
    printf("> GPU%d: %s %s unified addressing\n", 1, prop[1].name,
           (prop[1].unifiedAddressing ? "support" : "not support"));
    fflush(stdout);
    return iuva;
}

inline void calcIndex(int *haloStart, int *haloEnd, int *bodyStart,
                      int *bodyEnd, const int ngpus, const int iny)
{
    // for halo
    for (int i = 0; i < ngpus; i++)
    {
        if (i == 0 && ngpus == 2)
        {
            haloStart[i] = iny - NPAD2;
            haloEnd[i]   = iny - NPAD;

        }
        else
        {
            haloStart[i] = NPAD;
            haloEnd[i]   = NPAD2;
        }
    }

    // for body
    for (int i = 0; i < ngpus; i++)
    {
        if (i == 0 && ngpus == 2)
        {
            bodyStart[i] = NPAD;
            bodyEnd[i]   = iny - NPAD2;
        }
        else
        {
            bodyStart[i] = NPAD + NPAD;
            bodyEnd[i]   = iny - NPAD;
        }
    }
}

inline void calcSkips(int *src_skip, int *dst_skip, const int nx,
                      const int iny)
{
    src_skip[0] = nx * (iny - NPAD2);
    dst_skip[0] = 0;
    src_skip[1] = NPAD * nx;
    dst_skip[1] = (iny - NPAD) * nx;
}

// wavelet
__global__ void kernel_add_wavelet ( float *g_u2, float wavelets, const int nx,
                                     const int ny, const int ngpus)
{
    // global grid idx for (x,y) plane
    int ipos = (ngpus == 2 ? ny - 10 : ny / 2 - 10);
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = ipos * nx + ix;

    if(ix == nx / 2) g_u2[idx] += wavelets;
}

// fd kernel function
__global__ void kernel_2dfd_last(float *g_u1, float *g_u2, const int nx,
                                 const int iStart, const int iEnd)
{
    // global to slice : global grid idx for (x,y) plane
    unsigned int ix  = blockIdx.x * blockDim.x + threadIdx.x;

    // smem idx for current point
    unsigned int stx = threadIdx.x + NPAD;
    unsigned int idx  = ix + iStart * nx;

    // shared memory for u2 with size [4+16+4][4+16+4]
    __shared__ float tile[BDIMX + NPAD2];

    const float alpha = 0.12f;

    // register for y value
    float yval[9];

    for (int i = 0; i < 8; i++) yval[i] = g_u2[idx + (i - 4) * nx];

    // to be used in z loop
    int iskip = NPAD * nx;

#pragma unroll 9
    for (int iy = iStart; iy < iEnd; iy++)
    {
        // get front3 here
        yval[8] = g_u2[idx + iskip];

        if(threadIdx.x < NPAD)
        {
            tile[threadIdx.x]  = g_u2[idx - NPAD];
            tile[stx + BDIMX]    = g_u2[idx + BDIMX];
        }

        tile[stx] = yval[4];
        __syncthreads();

        if ( (ix >= NPAD) && (ix < nx - NPAD) )
        {
            // 8rd fd operator
            float tmp = coef[0] * tile[stx] * 2.0f;

#pragma unroll
            for(int d = 1; d <= 4; d++)
            {
                tmp += coef[d] * (tile[stx - d] + tile[stx + d]);
            }

#pragma unroll
            for(int d = 1; d <= 4; d++)
            {
                tmp += coef[d] * (yval[4 - d] + yval[4 + d]);
            }

            // time dimension
            g_u1[idx] = yval[4] + yval[4] - g_u1[idx] + alpha * tmp;
        }

#pragma unroll 8
        for (int i = 0; i < 8 ; i++)
        {
            yval[i] = yval[i + 1];
        }

        // advancd on global idx
        idx  += nx;
        __syncthreads();
    }
}

__global__ void kernel_2dfd(float *g_u1, float *g_u2, const int nx,
                            const int iStart, const int iEnd)
{
    // global to line index
    unsigned int ix  = blockIdx.x * blockDim.x + threadIdx.x;

    // smem idx for current point
    unsigned int stx = threadIdx.x + NPAD;
    unsigned int idx  = ix + iStart * nx;

    // shared memory for x dimension
    __shared__ float line[BDIMX + NPAD2];

    // a coefficient related to physical properties
    const float alpha = 0.12f;

    // register for y value
    float yval[9];

    for (int i = 0; i < 8; i++) yval[i] = g_u2[idx + (i - 4) * nx];

    // skip for the bottom most y value
    int iskip = NPAD * nx;

#pragma unroll 9
    for (int iy = iStart; iy < iEnd; iy++)
    {
        // get yval[8] here
        yval[8] = g_u2[idx + iskip];

        // read halo part
        if(threadIdx.x < NPAD)
        {
            line[threadIdx.x]  = g_u2[idx - NPAD];
            line[stx + BDIMX]    = g_u2[idx + BDIMX];
        }

        line[stx] = yval[4];
        __syncthreads();

        // 8rd fd operator
        if ( (ix >= NPAD) && (ix < nx - NPAD) )
        {
            // center point
            float tmp = coef[0] * line[stx] * 2.0f;

#pragma unroll
            for(int d = 1; d <= 4; d++)
            {
                tmp += coef[d] * ( line[stx - d] + line[stx + d]);
            }

#pragma unroll
            for(int d = 1; d <= 4; d++)
            {
                tmp += coef[d] * (yval[4 - d] + yval[4 + d]);
            }

            // time dimension
            g_u1[idx] = yval[4] + yval[4] - g_u1[idx] + alpha * tmp;
        }

#pragma unroll 8
        for (int i = 0; i < 8 ; i++)
        {
            yval[i] = yval[i + 1];
        }

        // advancd on global idx
        idx  += nx;
        __syncthreads();
    }
}

int main( int argc, char *argv[] )
{
    int ngpus;

    // check device count
    CHECK(cudaGetDeviceCount(&ngpus));
    printf("> CUDA-capable device count: %i\n", ngpus);

    // check p2p capability
    isCapableP2P(ngpus);
    isUnifiedAddressing(ngpus);

    //  get it from command line
    if (argc > 1)
    {
        if (atoi(argv[1]) > ngpus)
        {
            fprintf(stderr, "Invalid number of GPUs specified: %d is greater "
                    "than the total number of GPUs in this platform (%d)\n",
                    atoi(argv[1]), ngpus);
            exit(1);
        }

        ngpus  = atoi(argv[1]);
    }

    int iMovie = 10000;

    if(argc >= 3) iMovie = atoi(argv[2]);

    printf("> run with device: %i\n", ngpus);

    // size
    const int nsteps  = 600;
    const int nx      = 512;
    const int ny      = 512;
    const int iny     = ny / ngpus + NPAD * (ngpus - 1);

    size_t isize = nx * iny;
    size_t ibyte = isize * sizeof(float);
    size_t iexchange = NPAD * nx * sizeof(float);

    // set up gpu card
    float *d_u2[ngpus], *d_u1[ngpus];

    for(int i = 0; i < ngpus; i++)
    {
        // set device
        CHECK(cudaSetDevice(i));

        // allocate device memories
        CHECK(cudaMalloc ((void **) &d_u1[i], ibyte));
        CHECK(cudaMalloc ((void **) &d_u2[i], ibyte));

        CHECK(cudaMemset (d_u1[i], 0, ibyte));
        CHECK(cudaMemset (d_u2[i], 0, ibyte));

        printf("GPU %i: allocated %.2f MB gmem\n", i,
               (4.f * ibyte) / (1024.f * 1024.f) );
        setup_coef ();
    }

    // stream definition
    cudaStream_t stream_halo[ngpus], stream_body[ngpus];

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamCreate( &stream_halo[i] ));
        CHECK(cudaStreamCreate( &stream_body[i] ));
    }

    // calculate index for computation
    int haloStart[ngpus], bodyStart[ngpus], haloEnd[ngpus], bodyEnd[ngpus];
    calcIndex(haloStart, haloEnd, bodyStart, bodyEnd, ngpus, iny);

    int src_skip[ngpus], dst_skip[ngpus];

    if(ngpus > 1) calcSkips(src_skip, dst_skip, nx, iny);

    // kernel launch configuration
    dim3 block(BDIMX);
    dim3 grid(nx / block.x);

    // set up event for timing
    CHECK(cudaSetDevice(0));
    cudaEvent_t start, stop;
    CHECK (cudaEventCreate(&start));
    CHECK (cudaEventCreate(&stop ));
    CHECK(cudaEventRecord( start, 0 ));

    // main loop for wave propagation
    for(int istep = 0; istep < nsteps; istep++)
    {
        // save snap image
        if(iMovie == istep) saveSnapshotIstep(istep, nx, ny, ngpus, d_u2);

        // add wavelet only onto gpu0
        if (istep == 0)
        {
            CHECK(cudaSetDevice(0));
            kernel_add_wavelet<<<grid, block>>>(d_u2[0], 20.0, nx, iny, ngpus);
        }

        // halo part
        for (int i = 0; i < ngpus; i++)
        {
            CHECK(cudaSetDevice(i));

            // compute halo
            kernel_2dfd<<<grid, block, 0, stream_halo[i]>>>(d_u1[i], d_u2[i],
                    nx, haloStart[i], haloEnd[i]);

            // compute internal
            kernel_2dfd<<<grid, block, 0, stream_body[i]>>>(d_u1[i], d_u2[i],
                    nx, bodyStart[i], bodyEnd[i]);
        }

        // exchange halo
        if (ngpus > 1)
        {
            CHECK(cudaMemcpyAsync(d_u1[1] + dst_skip[0], d_u1[0] + src_skip[0],
                        iexchange, cudaMemcpyDefault, stream_halo[0]));
            CHECK(cudaMemcpyAsync(d_u1[0] + dst_skip[1], d_u1[1] + src_skip[1],
                        iexchange, cudaMemcpyDefault, stream_halo[1]));
        }

        for (int i = 0; i < ngpus; i++)
        {
            CHECK(cudaSetDevice(i));
            CHECK(cudaDeviceSynchronize());

            float *tmpu0 = d_u1[i];
            d_u1[i] = d_u2[i];
            d_u2[i] = tmpu0;
        }
    }

    CHECK(cudaSetDevice( 0 ));
    CHECK(cudaEventRecord( stop, 0 ));

    CHECK(cudaDeviceSynchronize());
    CHECK (cudaGetLastError());

    float elapsed_time_ms = 0.0f;
    CHECK (cudaEventElapsedTime( &elapsed_time_ms, start, stop ));

    elapsed_time_ms /= nsteps;
    printf("gputime: %8.2fms ", elapsed_time_ms);
    printf("performance: %8.2f MCells/s\n",
           (double) nx * ny / (elapsed_time_ms * 1e3f) );
    fflush(stdout);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // clear
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        CHECK (cudaStreamDestroy( stream_halo[i] ));
        CHECK (cudaStreamDestroy( stream_body[i] ));

        CHECK (cudaFree (d_u1[i]));
        CHECK (cudaFree (d_u2[i]));

        CHECK(cudaDeviceReset());
    }

    exit(EXIT_SUCCESS);
}
