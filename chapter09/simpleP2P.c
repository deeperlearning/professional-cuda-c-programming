#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <semaphore.h>
#include <cuda_runtime_api.h>

/*
 * A simple example of using the MPI and CUDA communication APIs to manually
 * transfer data from a GPU managed in one MPI process to a GPU managed in
 * another. The general steps performed are GPU0 -> cudaMemcpy -> rank0 ->
 * MPI_Isend -> rank1 -> cudaMemcpy -> GPU1.
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
#define SKIP_LARGE  10
#define LARGE_MESSAGE_SIZE  8192

int loop = LOOP_LARGE;

void initalData (void * sbuf, void * rbuf, size_t size)
{
    memset(sbuf, 'a', size);
    memset(rbuf, 'b', size);
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

    if (nprocs != 2)
    {
        if(rank == 0) printf("This test requires exactly two processes\n");

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    int other_proc = (rank == 1 ? 0 : 1);

    // Hard code GPU affinity since this example only works with 2 GPUs.
    int igpu = (rank == 1 ? 0 : 1);

    if(rank == 0 )
        printf("%s allocates %d MB pinned memory with regual mpi and "
               "bidirectional bandwidth\n", argv[0],
               MAX_MSG_SIZE / 1024 / 1024);

    printf("node=%d(%s): my other _proc = %d and using GPU=%d\n", rank,
            processor, other_proc, igpu);

    char *h_src, *h_rcv;
    CHECK(cudaSetDevice(igpu));
    CHECK(cudaMallocHost((void**)&h_src, MYBUFSIZE));
    CHECK(cudaMallocHost((void**)&h_rcv, MYBUFSIZE));

    char *d_src, *d_rcv;
    CHECK(cudaSetDevice(igpu));
    CHECK(cudaMalloc((void **)&d_src, MYBUFSIZE));
    CHECK(cudaMalloc((void **)&d_rcv, MYBUFSIZE));

    initalData(h_src, h_rcv, MYBUFSIZE);

    CHECK(cudaMemcpy(d_src, h_src, MYBUFSIZE, cudaMemcpyDefault));
    CHECK(cudaMemcpy(d_rcv, h_rcv, MYBUFSIZE, cudaMemcpyDefault));

    // latency test
    for(int size = 1024; size <= MAX_MSG_SIZE; size = size * 4)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0)
        {
            tstart = MPI_Wtime();

            for(int i = 0; i < loop; i++)
            {
                /*
                 * Transfer data from the GPU to the host to be transmitted to
                 * the other MPI process.
                 */
                CHECK(cudaMemcpy(h_src, d_src, size, cudaMemcpyDeviceToHost));

                // bi-directional transmission
                MPI_Irecv(h_rcv, size, MPI_CHAR, other_proc, 10, MPI_COMM_WORLD,
                          &recv_request);
                MPI_Isend(h_src, size, MPI_CHAR, other_proc, 100,
                          MPI_COMM_WORLD, &send_request);

                MPI_Waitall(1, &recv_request, &reqstat);
                MPI_Waitall(1, &send_request, &reqstat);

                /*
                 * Transfer the data received from the other MPI process to
                 * the device.
                 */
                CHECK(cudaMemcpy(d_rcv, h_rcv, size, cudaMemcpyHostToDevice));
            }

            tend = MPI_Wtime();
        }
        else
        {
            for(int i = 0; i < loop; i++)
            {
                /*
                 * Transfer data from the GPU to the host to be transmitted to
                 * the other MPI process.
                 */
                CHECK(cudaMemcpy(h_src, d_src, size, cudaMemcpyDeviceToHost));

                // bi-directional transmission
                MPI_Irecv(h_rcv, size, MPI_CHAR, other_proc, 100,
                          MPI_COMM_WORLD, &recv_request);
                MPI_Isend(h_src, size, MPI_CHAR, other_proc, 10, MPI_COMM_WORLD,
                          &send_request);

                MPI_Waitall(1, &recv_request, &reqstat);
                MPI_Waitall(1, &send_request, &reqstat);

                /*
                 * Transfer the data received from the other MPI process to
                 * the device.
                 */
                CHECK(cudaMemcpy(d_rcv, h_rcv, size, cudaMemcpyHostToDevice));
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0)
        {
            double latency = (tend - tstart) * 1e6 / (2.0 * loop);
            float performance = (float) size / (float) latency;
            printf("%6d %s %10.2f μs %10.2f MB/sec\n",
                   (size >= 1024 * 1024) ? size / 1024 / 1024 : size / 1024,
                   (size >= 1024 * 1024) ? "MB" : "KB", latency, performance);

            fflush(stdout);
        }
    }

    CHECK(cudaFreeHost(h_src));
    CHECK(cudaFreeHost(h_rcv));

    CHECK(cudaSetDevice(igpu));
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_rcv));

    MPI_Finalize();

    return EXIT_SUCCESS;
}
