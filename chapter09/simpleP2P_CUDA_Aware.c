#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

/*
 * An example of using a CUDA-aware MPI implementation to transfer an array
 * directly from one GPU to another, between MPI processes. Note that no CUDA
 * transfer API calls are used here, and that device pointers are passed
 * directly to MPI_Isend and MPI_Irecv.
 */

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define MESSAGE_ALIGNMENT 64
#define MAX_MSG_SIZE (1<<22)
#define MYBUFSIZE MAX_MSG_SIZE

#define LOOP_LARGE  100
#define FIELD_WIDTH 20
#define FLOAT_PRECISION 2

void SetDeviceBeforeInit()
{
    int devCount = 0;
    int rank = atoi(getenv("MV2_COMM_WORLD_RANK"));
    int idev = (rank == 0 ? 1 : 0);
    CHECK(cudaSetDevice(idev));

    printf("local rank=%d: and idev %d\n", rank, idev);
}

int main (int argc, char *argv[])
{
    int rank, nprocs, ilen;
    char processor[MPI_MAX_PROCESSOR_NAME];
    double tstart = 0.0, tend = 0.0;

    MPI_Status reqstat;
    MPI_Request send_request;
    MPI_Request recv_request;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Get_processor_name(processor, &ilen);

    if(nprocs != 2)
    {
        if(rank == 0) printf("This test requires exactly two processes\n");

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    char *h_src, *h_rcv;

    int other_proc = (rank == 1 ? 0 : 1);
    int igpu = (rank == 1 ? 0 : 1);

    int loop = LOOP_LARGE;

    printf("node=%d(%s): my other _proc = %d and using GPU=%d loop %d\n", rank,
           processor, other_proc, igpu, loop);

    char *d_src, *d_rcv;
    CHECK(cudaSetDevice(igpu));
    CHECK(cudaMalloc((void **)&d_src, MYBUFSIZE));
    CHECK(cudaMalloc((void **)&d_rcv, MYBUFSIZE));

    for (int size = 1; size <= MAX_MSG_SIZE; size *= 2)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        CHECK(cudaMemset(d_src, 'a', size));
        CHECK(cudaMemset(d_rcv, 'b', size));

        if(rank == 0)
        {
            tstart = MPI_Wtime();

            for(int i = 0; i < loop; i++)
            {
                MPI_Isend(d_src, size, MPI_CHAR, other_proc, 100,
                        MPI_COMM_WORLD, &send_request);
                MPI_Irecv(d_rcv, size, MPI_CHAR, other_proc, 10, MPI_COMM_WORLD,
                        &recv_request);

                MPI_Waitall(1, &recv_request, &reqstat);
                MPI_Waitall(1, &send_request, &reqstat);

            }

            tend = MPI_Wtime();
        }
        else
        {
            for(int i = 0; i < loop; i++)
            {
                MPI_Isend(d_src, size, MPI_CHAR, other_proc, 10, MPI_COMM_WORLD,
                        &send_request);
                MPI_Irecv(d_rcv, size, MPI_CHAR, other_proc, 100,
                        MPI_COMM_WORLD, &recv_request);

                MPI_Waitall(1, &recv_request, &reqstat);
                MPI_Waitall(1, &send_request, &reqstat);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0)
        {
            double tmp = size / 1e6 * loop  * 2;
            double t = (tend - tstart);

            printf("%-*d%*.*f\n", 10, size, FIELD_WIDTH, FLOAT_PRECISION,
                    tmp / t);
            fflush(stdout);
        }
    }

    CHECK(cudaSetDevice(igpu));
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_rcv));

    MPI_Finalize();

    return EXIT_SUCCESS;
}
