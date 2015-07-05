#include "../common/common.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

/*
 * This example demonstrates P2P ping-ponging of data from one GPU to another,
 * within the same node. By enabling peer-to-peer transfers, you ensure that
 * copies between GPUs go directly over the PCIe bus. If P2P is not enabled,
 * host memory must be used as a staging area for GPU-to-GPU cudaMemcpys.
 */

__global__ void iKernel(float *src, float *dst)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] * 2.0f;
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
    }

    if(iCount != ngpus)
    {
        printf("> no enough device to run this application\n");
    }

    return (iCount == ngpus);
}

/*
 * enable P2P memcopies between GPUs (all GPUs must be compute capability 2.0 or
 * later (Fermi or later)).
 */
inline void enableP2P (int ngpus)
{
    for( int i = 0; i < ngpus; i++ )
    {
        CHECK(cudaSetDevice(i));

        for(int j = 0; j < ngpus; j++)
        {
            if(i == j) continue;

            int peer_access_available = 0;
            CHECK(cudaDeviceCanAccessPeer(&peer_access_available, i, j));

            if (peer_access_available)
            {
                CHECK(cudaDeviceEnablePeerAccess(j, 0));
                printf("> GPU%d enabled direct access to GPU%d\n", i, j);
            }
            else
            {
                printf("(%d, %d)\n", i, j );
            }
        }
    }
}

inline void disableP2P (int ngpus)
{
    for( int i = 0; i < ngpus; i++ )
    {
        CHECK(cudaSetDevice(i));

        for(int j = 0; j < ngpus; j++)
        {
            if( i == j ) continue;

            int peer_access_available = 0;
            CHECK(cudaDeviceCanAccessPeer( &peer_access_available, i, j) );

            if( peer_access_available )
            {
                CHECK(cudaDeviceDisablePeerAccess(j));
                printf("> GPU%d disabled direct access to GPU%d\n", i, j);
            }
        }
    }
}

void initialData(float *ip, int size)
{
    for(int i = 0; i < size; i++)
    {
        ip[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main(int argc, char **argv)
{
    int ngpus;

    // check device count
    CHECK(cudaGetDeviceCount(&ngpus));
    printf("> CUDA-capable device count: %i\n", ngpus);

    // check p2p capability
    isCapableP2P(ngpus);

    // get ngpus from command line
    if (argc > 1)
    {
        if (atoi(argv[1]) > ngpus)
        {
            fprintf(stderr, "Invalid number of GPUs specified: %d is greater "
                    "than the total number of GPUs in this platform (%d)\n",
                    atoi(argv[1]), ngpus);
            return 1;
        }

        ngpus  = atoi(argv[1]);
    }

    if (ngpus > 2)
    {
        fprintf(stderr, "No more than 2 GPUs supported\n");
        return 1;
    }

    if (ngpus > 1) enableP2P(ngpus);

    // Allocate buffers
    int iSize = 1024 * 1024 * 16;
    const size_t iBytes = iSize * sizeof(float);
    printf("\nAllocating buffers (%iMB on each GPU and CPU Host)...\n",
           int(iBytes / 1024 / 1024));

    float **d_src = (float **)malloc(sizeof(float) * ngpus);
    float **d_rcv = (float **)malloc(sizeof(float) * ngpus);
    float **h_src = (float **)malloc(sizeof(float) * ngpus);
    cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * ngpus);

    // Create CUDA event handles
    cudaEvent_t start, stop;
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc(&d_src[i], iBytes));
        CHECK(cudaMalloc(&d_rcv[i], iBytes));
        CHECK(cudaMallocHost((void **) &h_src[i], iBytes));

        CHECK(cudaStreamCreate(&stream[i]));
    }

    for (int i = 0; i < ngpus; i++)
    {
        initialData(h_src[i], iSize);
    }

    // unidirectional gmem copy
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventRecord(start, 0));

    for (int i = 0; i < 100; i++)
    {
        if (i % 2 == 0)
        {
            CHECK(cudaMemcpy(d_src[1], d_src[0], iBytes,
                        cudaMemcpyDeviceToDevice));
        }
        else
        {
            CHECK(cudaMemcpy(d_src[0], d_src[1], iBytes,
                        cudaMemcpyDeviceToDevice));
        }
    }

    CHECK(cudaSetDevice(0));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    float elapsed_time_ms;
    CHECK(cudaEventElapsedTime(&elapsed_time_ms, start, stop ));

    elapsed_time_ms /= 100.0f;
    printf("Ping-pong unidirectional cudaMemcpy:\t\t %8.2f ms ",
           elapsed_time_ms);
    printf("performance: %8.2f GB/s\n",
            (float)iBytes / (elapsed_time_ms * 1e6f));

    //  bidirectional asynchronous gmem copy
    CHECK(cudaEventRecord(start, 0));

    for (int i = 0; i < 100; i++)
    {
        CHECK(cudaMemcpyAsync(d_src[1], d_src[0], iBytes,
                    cudaMemcpyDeviceToDevice, stream[0]));
        CHECK(cudaMemcpyAsync(d_rcv[0], d_rcv[1], iBytes,
                    cudaMemcpyDeviceToDevice, stream[1]));
    }

    CHECK(cudaSetDevice(0));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    elapsed_time_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&elapsed_time_ms, start, stop ));

    elapsed_time_ms /= 100.0f;
    printf("Ping-pong bidirectional cudaMemcpyAsync:\t %8.2fms ",
           elapsed_time_ms);
    printf("performance: %8.2f GB/s\n",
           (float) 2.0f * iBytes / (elapsed_time_ms * 1e6f) );

    disableP2P(ngpus);

    // free
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaFree(d_src[i]));
        CHECK(cudaFree(d_rcv[i]));
        CHECK(cudaStreamDestroy(stream[i]));
        CHECK(cudaDeviceReset());
    }

    exit(EXIT_SUCCESS);
}
