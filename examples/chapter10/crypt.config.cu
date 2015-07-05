#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

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

typedef enum { ENCRYPT, DECRYPT } action;

__constant__ int dkey[KEY_LENGTH];

/*
 * doCrypt implements the core logic of IDEA. It iterates over the byte
 * chunks stored in plainList and outputs their encrypted/decrypted form to the
 * corresponding element in cryptList using the secret key provided.
 */
__device__ void doCrypt(int chunk, signed char *plain, signed char *crypt)
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
        x1 = (int)((((long)x1 * dkey[ik++]) % 0x10001L) & 0xffff);
        x2 = ((x2 + dkey[ik++]) & 0xffff);
        x3 = ((x3 + dkey[ik++]) & 0xffff);
        x4 = (int)((((long)x4 * dkey[ik++]) % 0x10001L) & 0xffff);

        t2 = (x1 ^ x3);
        t2 = (int)((((long)t2 * dkey[ik++]) % 0x10001L) & 0xffff);

        t1 = ((t2 + (x2 ^ x4)) & 0xffff);
        t1 = (int)((((long)t1 * dkey[ik++]) % 0x10001L) & 0xffff);
        t2 = (t1 + t2 & 0xffff);

        x1 = (x1 ^ t1);
        x4 = (x4 ^ t2);
        t2 = (t2 ^ x2);
        x2 = (x3 ^ t1);
        x3 = t2;
    }
    while(--r != 0);

    x1 = (int)((((long)x1 * dkey[ik++]) % 0x10001L) & 0xffff);
    x3 = ((x3 + dkey[ik++]) & 0xffff);
    x2 = ((x2 + dkey[ik++]) & 0xffff);
    x4 = (int)((((long)x4 * dkey[ik++]) % 0x10001L) & 0xffff);

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

__global__ void encrypt_decrypt(signed char *plain, signed char *crypt,
                                int nChunks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    for ( ; tid < nChunks; tid += nthreads)
    {
        doCrypt(tid, plain, crypt);
    }
}

static void encrypt_decrypt_driver(signed char *plain, signed char *crypt,
                                   int *key,
                                   int plainLength, int nThreadsPerBlock)
{
    cudaEvent_t start, *finishes;
    cudaStream_t *streams;
    int nChunks, b, nBlocks;
    signed char *dPlain, *dCrypt;

    if (plainLength % CHUNK_SIZE != 0)
    {
        fprintf(stderr, "Invalid encryption: length of plain must be an even "
                "multiple of %d but is %d\n", CHUNK_SIZE, plainLength);
        exit(-1);
    }

    cudaDeviceProp info;
    CHECK(cudaGetDeviceProperties(&info, 0));
    nChunks = plainLength / CHUNK_SIZE;
    nBlocks = (nChunks + BLOCK_SIZE_IN_CHUNKS - 1) / BLOCK_SIZE_IN_CHUNKS;
    int nThreadBlocks = (nChunks + nThreadsPerBlock - 1) / nThreadsPerBlock;

    if (nThreadBlocks > info.maxGridSize[0])
    {
        nThreadBlocks = info.maxGridSize[0];
    }

    CHECK(cudaEventCreate(&start, 0));
    finishes = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * nBlocks);
    streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nBlocks);

    for (b = 0; b < nBlocks; b++)
    {
        CHECK(cudaStreamCreate(streams + b));
        CHECK(cudaEventCreate(finishes + b));
    }

    CHECK(cudaMalloc((void **)&dPlain,
                       plainLength * sizeof(signed char)));
    CHECK(cudaMalloc((void **)&dCrypt,
                       plainLength * sizeof(signed char)));

    CHECK(cudaEventRecord(start, streams[0]));
    CHECK(cudaMemcpyToSymbolAsync(dkey, key, KEY_LENGTH * sizeof(int), 0,
                    cudaMemcpyHostToDevice, streams[0]));
    CHECK(cudaStreamSynchronize(streams[0]));

    for (b = 0; b < nBlocks; b++)
    {
        int blockOffset = b * BLOCK_SIZE_IN_CHUNKS * CHUNK_SIZE;
        int localChunks = BLOCK_SIZE_IN_CHUNKS;

        if (b * BLOCK_SIZE_IN_CHUNKS + localChunks > nChunks)
        {
            localChunks = nChunks - b * BLOCK_SIZE_IN_CHUNKS;
        }

        CHECK(cudaMemcpyAsync(dPlain + blockOffset, plain + blockOffset,
                        localChunks * CHUNK_SIZE * sizeof(signed char),
                        cudaMemcpyHostToDevice, streams[b]));

        encrypt_decrypt<<<nThreadBlocks, nThreadsPerBlock, 0, streams[b]>>>(
            dPlain + blockOffset, dCrypt + blockOffset, localChunks);
        CHECK(cudaMemcpyAsync(crypt + blockOffset, dCrypt + blockOffset,
                        localChunks * CHUNK_SIZE * sizeof(signed char),
                        cudaMemcpyDeviceToHost, streams[b]));
        CHECK(cudaEventRecord(finishes[b], streams[b]));
    }

    CHECK(cudaDeviceSynchronize());

    float maxElapsed = 0.0;

    for (b = 0; b < nBlocks; b++)
    {
        float elapsed;
        CHECK(cudaEventElapsedTime(&elapsed, start, finishes[b]));
        maxElapsed = elapsed > maxElapsed ? elapsed : maxElapsed;
    }

    printf("Processed %d bytes in %f ms ( %f KB/ms )\n", plainLength,
           maxElapsed, ((float)plainLength / maxElapsed) / 1024.0f);

    for (b = 0; b < nBlocks; b++)
    {
        CHECK(cudaStreamDestroy(streams[b]));
        CHECK(cudaEventDestroy(finishes[b]));
    }

    free(streams);
    free(finishes);
    CHECK(cudaEventDestroy(start));

    CHECK(cudaFree(dPlain));
    CHECK(cudaFree(dCrypt));
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

    if (argc != 6)
    {
        printf("usage: %s <encrypt|decrypt> <file.in> <file.out> <key.file> "
               "<threads-per-block>\n", argv[0]);
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

    encrypt_decrypt_driver(text, crypt, key, textLen, nThreadsPerBlock);

    if (fwrite(crypt, sizeof(signed char), textLen, out) != textLen)
    {
        fprintf(stderr, "Failed writing crypt to %s\n", argv[3]);
        return (1);
    }

    fclose(out);

    cleanup(text, crypt, key, userkey);

    return (0);
}
