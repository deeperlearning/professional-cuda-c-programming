#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Recursive Implementation of Interleaved Pair Approach
float recursiveReduce(float *data, int const size)
{
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}

int main(int argc, char **argv)
{

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("%s starting reduction with array size %d\n", argv[0], size);

    // execution configuration
    int blocksize = 512;   // initial block size

    if(argc > 1)
    {
        blocksize = atoi(argv[1]);   // block size from command line argument
    }

    // allocate host memory
    size_t bytes = size * sizeof(float);
    float *h_idata = (float *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = (float)(rand() & 0xFF);
    }

    // cpu reduction
    double iStart = seconds();
    int cpu_sum = recursiveReduce (h_idata, size);
    double iElaps = seconds() - iStart;
    printf("cpu reduce elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // free host memory
    free(h_idata);

    return EXIT_SUCCESS;
}
