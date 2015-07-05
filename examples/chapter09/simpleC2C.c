#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/*
 * A simple example of using non-blocking communication between multiple MPI
 * processes to send and receive a char*. The sends and receives are done
 * repeatedly and timing results allows inter-process bandwidth to be
 * calculated.
 */

#define MESSAGE_ALIGNMENT 64
#define MAX_MSG_SIZE (1<<22)
#define MYBUFSIZE MAX_MSG_SIZE
#define LOOP_LARGE  100

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

    if(nprocs != 2)
    {
        if(rank == 0) printf("This test requires exactly two processes\n");

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    char *s_buf, *r_buf;
    s_buf = (char *)malloc(MYBUFSIZE);
    r_buf = (char *)malloc(MYBUFSIZE);

    int other_proc = (rank == 1 ? 0 : 1);

    if(rank == 0 )
    {
        printf("%s allocates %d MB dynamic memory aligned to 64 byte\n",
                              argv[0], MAX_MSG_SIZE / 1024 / 1024);
    }

    printf("node=%d(%s): my other _proc = %d\n", rank, processor, other_proc);

    int loop = LOOP_LARGE;

    // latency test
    for(int size = 1024; size <= MAX_MSG_SIZE; size = size * 4)
    {
        initalData(s_buf, r_buf, size);

        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0)
        {
            tstart = MPI_Wtime();

            for(int i = 0; i < loop; i++)
            {
                MPI_Irecv(r_buf, size, MPI_CHAR, other_proc, 10, MPI_COMM_WORLD,
                        &recv_request);
                MPI_Isend(s_buf, size, MPI_CHAR, other_proc, 100,
                        MPI_COMM_WORLD, &send_request);
                MPI_Waitall(1, &send_request, &reqstat);
                MPI_Waitall(1, &recv_request, &reqstat);
            }

            tend = MPI_Wtime();
        }
        else
        {
            for(int i = 0; i < loop; i++)
            {
                MPI_Irecv(r_buf, size, MPI_CHAR, other_proc, 100,
                        MPI_COMM_WORLD, &recv_request);
                MPI_Isend(s_buf, size, MPI_CHAR, other_proc, 10, MPI_COMM_WORLD,
                        &send_request);
                MPI_Waitall(1, &send_request, &reqstat);
                MPI_Waitall(1, &recv_request, &reqstat);
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

    free(s_buf);
    free(r_buf);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
