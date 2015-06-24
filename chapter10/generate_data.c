#include <stdlib.h>
#include <stdio.h>

/*
 * Generate a sample input for crypt to encrypt and decrypt. generate_data
 * allows the user to specify the output file and length in bytes.
 */

#define CHUNK_SIZE 1024
signed char chunk[CHUNK_SIZE];

int main(int argc, char **argv)
{
    int i, j;
    FILE *out;
    int outLength;
    int *ichunk;

    if (argc != 3)
    {
        printf("usage: %s <output-file> <output-file-length>\n", argv[0]);
        return (1);
    }

    out = fopen(argv[1], "w");

    if (out == NULL)
    {
        fprintf(stderr, "Failed opening %s for writing\n", argv[1]);
        return (1);
    }

    outLength = atoi(argv[2]);

    if (outLength % 8 != 0)
    {
        fprintf(stderr, "The specified length (%d) must be evenly divisible "
                "by 8\n", outLength);
        return (1);
    }

    // Write in chunks of CHUNK_SIZE.
    for (i = 0; i < outLength; i += CHUNK_SIZE)
    {
        int toWrite = CHUNK_SIZE;

        if (i + toWrite > outLength)
        {
            toWrite = outLength - i;
        }

        for (j = 0; j < toWrite; j++)
        {
            chunk[j] = (i * CHUNK_SIZE + j);
        }

        if (fwrite(chunk, 1, toWrite, out) != toWrite)
        {
            fprintf(stderr, "Error writing chunk of length %d\n", toWrite);
            return (1);
        }
    }

    fclose(out);

    return (0);
}
