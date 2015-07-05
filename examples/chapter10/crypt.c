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
// Length of the secret key, in bytes
#define USERKEY_LENGTH  8
#define BITS_PER_BYTE   8

typedef enum { ENCRYPT, DECRYPT } action;

// 8 bytes of binary data to be encrypted or decrypted in a single IDEA chunk.
typedef struct _encryptChunk
{
    signed char *data;
    struct _encryptChunk *next;
} encryptChunk;

/*
 * encrypt_decrypt implements the core logic of IDEA. It iterates over the byte
 * chunks stored in plainList and outputs their encrypted/decrypted form to the
 * corresponding element in cryptList using the secret key provided.
 */
static void encrypt_decrypt(encryptChunk *plainList, encryptChunk *cryptList,
                            int *key, int plainLength)
{
    if (plainLength % CHUNK_SIZE != 0)
    {
        fprintf(stderr, "Invalid encryption: length of plain must be an even "
                "multiple of %d but is %d\n", CHUNK_SIZE, plainLength);
        exit(1);
    }

    encryptChunk *currentPlain = plainList;
    encryptChunk *currentCrypt = cryptList;

    /*
     * Iterate over the 8-byte chunks in the plainList, encrypting/decrypting
     * each and storing the transformed bytes in the cryptList. Note that the
     * processing of each of these chunks is indepent from all other chunks and
     * that this is a computationally heavy algorithm.
     */
    while (currentPlain != NULL)
    {
        long x1, x2, x3, x4, t1, t2, ik, r;
        signed char *plain = currentPlain->data;
        signed char *crypt = currentCrypt->data;

        x1  = (((unsigned int)plain[0]) & 0xff);
        x1 |= ((((unsigned int)plain[1]) & 0xff) << BITS_PER_BYTE);
        x2  = (((unsigned int)plain[2]) & 0xff);
        x2 |= ((((unsigned int)plain[3]) & 0xff) << BITS_PER_BYTE);
        x3  = (((unsigned int)plain[4]) & 0xff);
        x3 |= ((((unsigned int)plain[5]) & 0xff) << BITS_PER_BYTE);
        x4  = (((unsigned int)plain[6]) & 0xff);
        x4 |= ((((unsigned int)plain[7]) & 0xff) << BITS_PER_BYTE);
        ik = 0;
        r = CHUNK_SIZE;

        do
        {
            x1 = (int)(((x1 * key[ik++]) % 0x10001L) & 0xffff);
            x2 = ((x2 + key[ik++]) & 0xffff);
            x3 = ((x3 + key[ik++]) & 0xffff);
            x4 = (int)(((x4 * key[ik++]) % 0x10001L) & 0xffff);

            t2 = (x1 ^ x3);
            t2 = (int)(((t2 * key[ik++]) % 0x10001L) & 0xffff);

            t1 = ((t2 + (x2 ^ x4)) & 0xffff);
            t1 = (int)(((t1 * key[ik++]) % 0x10001L) & 0xffff);
            t2 = (t1 + t2 & 0xffff);

            x1 = (x1 ^ t1);
            x4 = (x4 ^ t2);
            t2 = (t2 ^ x2);
            x2 = (x3 ^ t1);
            x3 = t2;
        }
        while(--r != 0);

        x1 = (int)(((x1 * key[ik++]) % 0x10001L) & 0xffff);
        x3 = ((x3 + key[ik++]) & 0xffff);
        x2 = ((x2 + key[ik++]) & 0xffff);
        x4 = (int)(((x4 * key[ik++]) % 0x10001L) & 0xffff);

        crypt[0] = (signed char) x1;
        crypt[1] = (signed char) ((unsigned long)x1 >> BITS_PER_BYTE);
        crypt[2] = (signed char) x3;
        crypt[3] = (signed char) ((unsigned long)x3 >> BITS_PER_BYTE);
        crypt[4] = (signed char) x2;
        crypt[5] = (signed char) ((unsigned long)x2 >> BITS_PER_BYTE);
        crypt[6] = (signed char) x4;
        crypt[7] = (signed char) ((unsigned long)x4 >> BITS_PER_BYTE);

        currentPlain = currentPlain->next;
        currentCrypt = currentCrypt->next;
    }
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
    int *key = (int *)malloc(sizeof(int) * KEY_LENGTH);

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
    int *key = (int *)malloc(sizeof(int) * KEY_LENGTH);
    int i, j, k;
    int t1, t2, t3;

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

    free(Z);

    return (key);
}

void readInputData(FILE *in, size_t textLen, encryptChunk **textHeadOut,
                   encryptChunk **textTailOut, encryptChunk **cryptHeadOut,
                   encryptChunk **cryptTailOut)
{
    size_t nread = 0;
    encryptChunk *textHead = NULL;
    encryptChunk *textTail = NULL;
    encryptChunk *cryptHead = NULL;
    encryptChunk *cryptTail = NULL;

    /*
     * Read input one chunk at a time into the linked list defined by textHead
     * and textTail. At the same time, pre-allocate output chunks into the
     * linked list defined by cryptHead and cryptTail.
     */
    while (nread < textLen)
    {
        signed char *textData = (signed char *)malloc(CHUNK_SIZE *
                                sizeof (signed char));
        int read = fread(textData, sizeof(signed char), CHUNK_SIZE, in);

        if (read != CHUNK_SIZE)
        {
            fprintf(stderr, "Failed reading text from input file\n");
            exit(1);
        }

        // Create input chunk
        encryptChunk *newChunk = (encryptChunk *)malloc(sizeof (encryptChunk));
        newChunk->data = textData;
        newChunk->next = NULL;

        if (textTail == NULL)
        {
            textHead = textTail = newChunk;
        }
        else
        {
            textTail->next = newChunk;
            textTail = newChunk;
        }

        // Create output chunk
        newChunk = (encryptChunk *)malloc(sizeof(encryptChunk));
        newChunk->data = (signed char *)malloc(CHUNK_SIZE *
                                               sizeof(signed char));
        newChunk->next = NULL;

        if (cryptTail == NULL)
        {
            cryptHead = cryptTail = newChunk;
        }
        else
        {
            cryptTail->next = newChunk;
            cryptTail = newChunk;
        }

        nread += read;
    }

    *textHeadOut = textHead;
    *textTailOut = textTail;
    *cryptHeadOut = cryptHead;
    *cryptTailOut = cryptTail;
}

void cleanupList(encryptChunk *l)
{
    while (l != NULL)
    {
        encryptChunk *next = l->next;
        free(l);
        l = next;
    }
}

void cleanup(encryptChunk *textList, encryptChunk *cryptList, int *key,
             int16_t *userkey)
{
    free(key);
    free(userkey);
    cleanupList(textList);
    cleanupList(cryptList);
}

/*
 * Initialize application state by reading inputs from the disk and
 * pre-allocating memory. Hand off to encrypt_decrypt to perform the actualy
 * encryption or decryption. Then, write the encrypted/decrypted results to
 * disk.
 */
int main(int argc, char **argv)
{
    encryptChunk *textHead = NULL;
    encryptChunk *textTail = NULL;
    encryptChunk *cryptHead = NULL;
    encryptChunk *cryptTail = NULL;
    FILE *in, *out, *keyfile;
    size_t textLen, keyFileLength;
    int16_t *userkey;
    int *key;
    action a;

    if (argc != 5)
    {
        printf("usage: %s <encrypt|decrypt> <file.in> <file.out> <key.file>\n",
               argv[0]);
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

    readInputData(in, textLen, &textHead, &textTail, &cryptHead, &cryptTail);
    fclose(in);

    encrypt_decrypt(textHead, cryptHead, key, textLen);

    encryptChunk *curr = cryptHead;

    while (curr != NULL)
    {
        if (fwrite(curr->data, sizeof(signed char), CHUNK_SIZE, out) !=
                CHUNK_SIZE)
        {
            fprintf(stderr, "Failed writing crypt to %s\n", argv[3]);
            return (1);
        }

        curr = curr->next;
    }

    fclose(out);

    cleanup(textHead, cryptHead, key, userkey);

    return (0);
}
