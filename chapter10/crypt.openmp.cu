#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

/*
 * The crypt application implements IDEA encryption and decryption of a single
 * input file using the secret key provided.
 */

// Chunking size for IDEA, in bytes
#define CHUNK_SIZE  8
// Length of the encryption/decryption keys, in bytes
#define KEY_LENGTH  52
#define BLOCK_SIZE_IN_CHUNKS    1024000
// Length of the secret key, in bytes
#define USERKEY_LENGTH  8
#define BITS_PER_BYTE   8

typedef struct _device_context
{
    signed char *dPlain, *dCrypt;
    cudaStream_t *streams;
    int nBlocks;
} device_context;

typedef enum { ENCRYPT, DECRYPT } action;

__constant__ int dkey[KEY_LENGTH];

/*
 * doCrypt implements the core logic of IDEA. It iterates over the byte
 * chunks stored in plainList and outputs their encrypted/decrypted form to the
 * corresponding element in cryptList using the secret key provided.
 */
__host__ __device__ void doCrypt(int chunk, signed char *plain,
                                 signed char *crypt, int *key)
{
    long x1, x2, x3, x4, t1, t2, ik, r;

    x1  = (((unsigned int)plain[chunk * CHUNK_SIZE]) & 0xff);
    x1 |= ((((unsigned int)plain[chunk * CHUNK_SIZE + 1]) & 0xff) <<
           BITS_PER_BYTE);
    x2  = (((unsigned int)plain[chunk * CHUNK_SIZE + 2]) & 0xff);
    x2 |= ((((unsigned int)plain[chunk * CHUNK_SIZE + 3]) & 0xff) <<
           BITS_PER_BYTE);
    x3  = (((unsigned int)plain[chunk * CHUNK_SIZE + 4]) & 0xff);
    x3 |= ((((unsigned int)plain[chunk * CHUNK_SIZE + 5]) & 0xff) <<
           BITS_PER_BYTE);
    x4  = (((unsigned int)plain[chunk * CHUNK_SIZE + 6]) & 0xff);
    x4 |= ((((unsigned int)plain[chunk * CHUNK_SIZE + 7]) & 0xff) <<
           BITS_PER_BYTE);
    ik  = 0;
    r = CHUNK_SIZE;

    do
    {
        x1 = (int)((((long)x1 * key[ik++]) % 0x10001L) & 0xffff);
        x2 = ((x2 + key[ik++]) & 0xffff);
        x3 = ((x3 + key[ik++]) & 0xffff);
        x4 = (int)((((long)x4 * key[ik++]) % 0x10001L) & 0xffff);

        t2 = (x1 ^ x3);
        t2 = (int)((((long)t2 * key[ik++]) % 0x10001L) & 0xffff);

        t1 = ((t2 + (x2 ^ x4)) & 0xffff);
        t1 = (int)((((long)t1 * key[ik++]) % 0x10001L) & 0xffff);
        t2 = (t1 + t2 & 0xffff);

        x1 = (x1 ^ t1);
        x4 = (x4 ^ t2);
        t2 = (t2 ^ x2);
        x2 = (x3 ^ t1);
        x3 = t2;
    }
    while(--r != 0);

    x1 = (int)((((long)x1 * key[ik++]) % 0x10001L) & 0xffff);
    x3 = ((x3 + key[ik++]) & 0xffff);
    x2 = ((x2 + key[ik++]) & 0xffff);
    x4 = (int)((((long)x4 * key[ik++]) % 0x10001L) & 0xffff);

    crypt[chunk * CHUNK_SIZE]     = (signed char) x1;
    crypt[chunk * CHUNK_SIZE + 1] = (signed char) ((unsigned long)x1 >>
                                    BITS_PER_BYTE);
    crypt[chunk * CHUNK_SIZE + 2] = (signed char) x3;
    crypt[chunk * CHUNK_SIZE + 3] = (signed char) ((unsigned long)x3 >>
                                    BITS_PER_BYTE);
    crypt[chunk * CHUNK_SIZE + 4] = (signed char) x2;
    crypt[chunk * CHUNK_SIZE + 5] = (signed char) ((unsigned long)x2 >>
                                    BITS_PER_BYTE);
    crypt[chunk * CHUNK_SIZE + 6] = (signed char) x4;
    crypt[chunk * CHUNK_SIZE + 7] = (signed char) ((unsigned long)x4 >>
                                    BITS_PER_BYTE);
}

__global__ void d_encrypt_decrypt(signed char *plain, signed char *crypt,
                                  int nChunks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    for ( ; tid < nChunks; tid += nthreads)
    {
        doCrypt(tid, plain, crypt, dkey);
    }
}

static void h_encrypt_decrypt(signed char *plain, signed char *crypt, int *key,
                              int plainLength)
{
    int c;
    int nChunks = plainLength / CHUNK_SIZE;
    #pragma omp parallel for firstprivate(nChunks) private(c)

    for (c = 0; c < nChunks; c++)
    {
        doCrypt(c, plain, crypt, key);
    }
}

static void init_context(device_context *ctx, int plainLength)
{
    signed char *dPlain, *dCrypt;
    cudaStream_t *streams;
    int nBlocks, b;

    if (plainLength % CHUNK_SIZE != 0)
    {
        fprintf(stderr, "Invalid encryption: length of plain must be an even "
                "multiple of %d but is %d\n", CHUNK_SIZE, plainLength);
        exit(-1);
    }

    CHECK(cudaMalloc((void **)&dPlain,
                       plainLength * sizeof(signed char)));
    CHECK(cudaMalloc((void **)&dCrypt,
                       plainLength * sizeof(signed char)));

    int nChunks = plainLength / CHUNK_SIZE;
    nBlocks = (nChunks + BLOCK_SIZE_IN_CHUNKS - 1) / BLOCK_SIZE_IN_CHUNKS;

    streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nBlocks);

    for (b = 0; b < nBlocks; b++)
    {
        CHECK(cudaStreamCreate(streams + b));
    }

    ctx->dPlain = dPlain;
    ctx->dCrypt = dCrypt;
    ctx->streams = streams;
    ctx->nBlocks = nBlocks;
}

static void encrypt_decrypt_driver(signed char *plain, signed char *crypt,
                                   int *key,
                                   int plainLength, int nThreadsPerBlock,
                                   device_context *ctx)
{
    int b;

    cudaDeviceProp info;
    CHECK(cudaGetDeviceProperties(&info, 0));
    int nChunks = plainLength / CHUNK_SIZE;
    int nThreadBlocks = (nChunks + nThreadsPerBlock - 1) / nThreadsPerBlock;

    if (nThreadBlocks > info.maxGridSize[0])
    {
        nThreadBlocks = info.maxGridSize[0];
    }

    CHECK(cudaMemcpyToSymbolAsync(dkey, key, KEY_LENGTH * sizeof(int), 0,
                    cudaMemcpyHostToDevice, (ctx->streams)[0]));
    CHECK(cudaStreamSynchronize((ctx->streams)[0]));

    for (b = 0; b < ctx->nBlocks; b++)
    {
        int blockOffset = b * BLOCK_SIZE_IN_CHUNKS * CHUNK_SIZE;
        int localChunks = BLOCK_SIZE_IN_CHUNKS;

        if (b * BLOCK_SIZE_IN_CHUNKS + localChunks > nChunks)
        {
            localChunks = nChunks - b * BLOCK_SIZE_IN_CHUNKS;
        }

        CHECK(cudaMemcpyAsync(ctx->dPlain + blockOffset, plain + blockOffset,
                        localChunks * CHUNK_SIZE * sizeof(signed char),
                        cudaMemcpyHostToDevice, (ctx->streams)[b]));

        d_encrypt_decrypt<<<nThreadBlocks, nThreadsPerBlock, 0,
                          (ctx->streams)[b] >>> (
                          ctx->dPlain + blockOffset, ctx->dCrypt + blockOffset,
                          localChunks);
        CHECK(cudaMemcpyAsync(crypt + blockOffset, ctx->dCrypt + blockOffset,
                        localChunks * CHUNK_SIZE * sizeof(signed char),
                        cudaMemcpyDeviceToHost, (ctx->streams)[b]));
    }
}

static void cleanup_context(device_context *ctx)
{
    int b;

    for (b = 0; b < ctx->nBlocks; b++)
    {
        CHECK(cudaStreamDestroy(ctx->streams[b]));
    }

    free(ctx->streams);

    CHECK(cudaFree(ctx->dPlain));
    CHECK(cudaFree(ctx->dCrypt));
}

/*
 * Get the length of a file on disk.
 */
static size_t getFileLength(FILE *fp)
{
    fseek(fp, 0L, SEEK_END);
    size_t fileLen = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    return (fileLen);
}

/*
 * inv is used to generate the key used for decryption from the secret key.
 */
static int inv(int x)
{
    int t0, t1;
    int q, y;

    if (x <= 1)             // Assumes positive x.
        return (x);          // 0 and 1 are self-inverse.

    t1 = 0x10001 / x;       // (2**16+1)/x; x is >= 2, so fits 16 bits.
    y = 0x10001 % x;

    if (y == 1)
        return ((1 - t1) & 0xffff);

    t0 = 1;

    do
    {
        q = x / y;
        x = x % y;
        t0 += q * t1;

        if (x == 1) return (t0);

        q = y / x;
        y = y % x;
        t1 += q * t0;
    }
    while (y != 1);

    return ((1 - t1) & 0xffff);
}

/*
 * Generate the key to be used for encryption, based on the user key read from
 * disk.
 */
static int *generateEncryptKey(int16_t *userkey)
{
    int i, j;
    int *key;

    CHECK(cudaMallocHost(&key, KEY_LENGTH * sizeof(int)));
    memset(key, 0x00, sizeof(int) * KEY_LENGTH);

    for (i = 0; i < CHUNK_SIZE; i++)
    {
        key[i] = (userkey[i] & 0xffff);
    }

    for (i = CHUNK_SIZE; i < KEY_LENGTH; i++)
    {
        j = i % CHUNK_SIZE;

        if (j < 6)
        {
            key[i] = ((key[i - 7] >> 9) | (key[i - 6] << 7))
                     & 0xffff;
            continue;
        }

        if (j == 6)
        {
            key[i] = ((key[i - 7] >> 9) | (key[i - 14] << 7))
                     & 0xffff;
            continue;
        }

        key[i] = ((key[i - 15] >> 9) | (key[i - 14] << 7))
                 & 0xffff;
    }

    return (key);
}

/*
 * Generate the key to be used for decryption, based on the user key read from
 * disk.
 */
static int *generateDecryptKey(int16_t *userkey)
{
    int *key;
    int i, j, k;
    int t1, t2, t3;

    CHECK(cudaMallocHost(&key, KEY_LENGTH * sizeof(int)));
    int *Z = generateEncryptKey(userkey);

    t1 = inv(Z[0]);
    t2 = - Z[1] & 0xffff;
    t3 = - Z[2] & 0xffff;

    key[51] = inv(Z[3]);
    key[50] = t3;
    key[49] = t2;
    key[48] = t1;

    j = 47;
    k = 4;

    for (i = 0; i < 7; i++)
    {
        t1 = Z[k++];
        key[j--] = Z[k++];
        key[j--] = t1;
        t1 = inv(Z[k++]);
        t2 = -Z[k++] & 0xffff;
        t3 = -Z[k++] & 0xffff;
        key[j--] = inv(Z[k++]);
        key[j--] = t2;
        key[j--] = t3;
        key[j--] = t1;
    }

    t1 = Z[k++];
    key[j--] = Z[k++];
    key[j--] = t1;
    t1 = inv(Z[k++]);
    t2 = -Z[k++] & 0xffff;
    t3 = -Z[k++] & 0xffff;
    key[j--] = inv(Z[k++]);
    key[j--] = t3;
    key[j--] = t2;
    key[j--] = t1;

    CHECK(cudaFreeHost(Z));

    return (key);
}

void readInputData(FILE *in, size_t textLen, signed char **text,
                   signed char **crypt)
{
    CHECK(cudaMallocHost(text, textLen * sizeof(signed char)));
    CHECK(cudaMallocHost(crypt, textLen * sizeof(signed char)));

    if (fread(*text, sizeof(signed char), textLen, in) != textLen)
    {
        fprintf(stderr, "Failed reading text from input file\n");
        exit(1);
    }
}

void cleanup(signed char *text, signed char *crypt, int *key,
             int16_t *userkey)
{
    free(userkey);
    CHECK(cudaFreeHost(key));
    CHECK(cudaFreeHost(text));
    CHECK(cudaFreeHost(crypt));
}

/*
 * Initialize application state by reading inputs from the disk and
 * pre-allocating memory. Hand off to encrypt_decrypt to perform the actualy
 * encryption or decryption. Then, write the encrypted/decrypted results to
 * disk.
 */
int main(int argc, char **argv)
{
    FILE *in, *out, *keyfile;
    signed char *text, *crypt;
    size_t textLen, keyFileLength;
    int16_t *userkey;
    int *key;
    action a;
    cudaEvent_t startEvent, finishEvent;

    if (argc != 8)
    {
        printf("usage: %s <encrypt|decrypt> <file.in> <file.out> <key.file> "
               "<threads-per-block> <ncpus> <cpu-percent>\n", argv[0]);
        return (1);
    }

    // Are we encrypting or decrypting?
    if (strncmp(argv[1], "encrypt", 7) == 0)
    {
        a = ENCRYPT;
    }
    else if (strncmp(argv[1], "decrypt", 7) == 0)
    {
        a = DECRYPT;
    }
    else
    {
        fprintf(stderr, "The action specified ('%s') is not valid. Must be "
                "either 'encrypt' or 'decrypt'\n", argv[1]);
        return (1);
    }

    // Input file
    in = fopen(argv[2], "r");

    if (in == NULL)
    {
        fprintf(stderr, "Unable to open %s for reading\n", argv[2]);
        return (1);
    }

    // Output file
    out = fopen(argv[3], "w");

    if (out == NULL)
    {
        fprintf(stderr, "Unable to open %s for writing\n", argv[3]);
        return (1);
    }

    // Key file
    keyfile = fopen(argv[4], "r");

    if (keyfile == NULL)
    {
        fprintf(stderr, "Unable to open key file %s for reading\n", argv[4]);
        return (1);
    }

    int nThreadsPerBlock = atoi(argv[5]);
    int ncpus = atoi(argv[6]);
    float cpu_percent = atof(argv[7]);
    omp_set_num_threads(ncpus);

    keyFileLength = getFileLength(keyfile);

    if (keyFileLength != sizeof(*userkey) * USERKEY_LENGTH)
    {
        fprintf(stderr, "Invalid user key file length %lu, must be %lu\n",
                keyFileLength, sizeof(*userkey) * USERKEY_LENGTH);
        return (1);
    }

    userkey = (int16_t *)malloc(sizeof(int16_t) * USERKEY_LENGTH);

    if (userkey == NULL)
    {
        fprintf(stderr, "Error allocating user key\n");
        return (1);
    }

    if (fread(userkey, sizeof(*userkey), USERKEY_LENGTH, keyfile) !=
            USERKEY_LENGTH)
    {
        fprintf(stderr, "Error reading user key\n");
        return (1);
    }

    if (a == ENCRYPT)
    {
        key = generateEncryptKey(userkey);
    }
    else
    {
        key = generateDecryptKey(userkey);
    }

    textLen = getFileLength(in);

    if (textLen % CHUNK_SIZE != 0)
    {
        fprintf(stderr, "Invalid input file length %lu, must be evenly "
                "divisible by %d\n", textLen, CHUNK_SIZE);
        return (1);
    }

    readInputData(in, textLen, &text, &crypt);
    fclose(in);

    int nDevices;

    if (cpu_percent >= 1.0f ||
            cudaGetDeviceCount(&nDevices) == cudaErrorNoDevice)
    {

        // If no devices are found, run all computation on the CPU using OpenMP.
        double overall_start = seconds();
        h_encrypt_decrypt(text, crypt, key, textLen);
        double overall_finish = seconds();
        double overall_ms = 1000.0 * (overall_finish - overall_start);
        printf("Processed %d bytes in %.3f ms on CPU ( %.4f KB/s )\n",
                textLen, overall_ms,
                ((float)textLen / overall_ms) / 1024.0f);
    }
    else
    {
        int d;

        int nTotalChunks = textLen / CHUNK_SIZE;
        int nCpuChunks = nTotalChunks * cpu_percent;

        if (nCpuChunks > nTotalChunks) nCpuChunks = nTotalChunks;

        int nGpuChunks = nTotalChunks - nCpuChunks;
        int gpuLen = nGpuChunks * CHUNK_SIZE;

        printf("Processing %d bytes on the GPUs, %d bytes on the CPU\n", gpuLen,
                textLen - gpuLen);

        int chunksPerGpu = (nGpuChunks + nDevices - 1) / nDevices;
        device_context *ctxs = (device_context *)malloc(nDevices *
                sizeof(device_context));

        for (d = 0; d < nDevices; d++)
        {
            CHECK(cudaSetDevice(d));
            int start = d * chunksPerGpu * CHUNK_SIZE;
            int len = chunksPerGpu * CHUNK_SIZE;

            if (start + len > gpuLen)
            {
                len = gpuLen - start;
            }

            init_context(ctxs + d, len);
        }

        CHECK(cudaEventCreate(&startEvent));
        CHECK(cudaEventCreate(&finishEvent));

        /*
         * Iterate over each device, launching a subset of the total chunks at
         * a time.
         */
        double overall_start = seconds();
        CHECK(cudaEventRecord(startEvent));

        for (d = 0; d < nDevices; d++)
        {
            CHECK(cudaSetDevice(d));
            int start = d * chunksPerGpu * CHUNK_SIZE;
            int len = chunksPerGpu * CHUNK_SIZE;

            if (start + len > gpuLen)
            {
                len = gpuLen - start;
            }

            encrypt_decrypt_driver(text + start, crypt + start, key, len,
                    nThreadsPerBlock, ctxs + d);
        }

        CHECK(cudaEventRecord(finishEvent));

        double cpu_start = seconds();
        int cpuStart = gpuLen;
        h_encrypt_decrypt(text + cpuStart, crypt + cpuStart, key, textLen -
                cpuStart);
        double cpu_finish = seconds();
        double cpu_overall_ms = 1000.0 * (cpu_finish - cpu_start);

        // Wait for each device to finish its work.
        for (d = 0; d < nDevices; d++)
        {
            CHECK(cudaSetDevice(d));
            CHECK(cudaDeviceSynchronize());
        }

        double overall_finish = seconds();

        for (d = 0; d < nDevices; d++)
        {
            // Clean up any CUDA resource allocated for this device.
            CHECK(cudaSetDevice(d));
            cleanup_context(ctxs + d);
        }

        float gpuElapsed;
        CHECK(cudaEventElapsedTime(&gpuElapsed, startEvent, finishEvent));
        printf("Processed %d bytes in %.3f ms on GPUs ( %.4f KB/ms )\n",
                gpuLen, gpuElapsed, ((float)gpuLen / gpuElapsed) / 1024.0f);
        printf("Processed %d bytes in %.3f ms on CPU ( %.4f KB/ms )\n",
                textLen - cpuStart, cpu_overall_ms,
                ((float)(textLen - cpuStart) / cpu_overall_ms) / 1024.0f);

        // Display the aggregate performance of all devices.
        double overall_elapsed_ms = 1000.0 * (overall_finish - overall_start);
        printf("In total, processed %d bytes in %.3f ms on %d devices and the "
                "CPU\n", textLen, overall_elapsed_ms, nDevices);
        printf("Aggregate bandwith = %f KB/ms\n",
                (float)(textLen / 1024) / overall_elapsed_ms);
        free(ctxs);
    }

    if (fwrite(crypt, sizeof(signed char), textLen, out) != textLen)
    {
        fprintf(stderr, "Failed writing crypt to %s\n", argv[3]);
        return (1);
    }

    fclose(out);

    cleanup(text, crypt, key, userkey);

    return (0);
}
