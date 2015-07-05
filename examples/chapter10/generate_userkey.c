#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/*
 * generate_userkey generates a 64-bit secret key.
 */
#define SECRET_KEY_LENGTH   8

int main(int argc, char **argv)
{
    int i;
    int16_t *userkey;
    FILE *out;

    if (argc != 2)
    {
        printf("usage: %s <output-file>\n", argv[0]);
        return 1;
    }

    out = fopen(argv[1], "w");

    if (out == NULL)
    {
        fprintf(stderr, "Error opening %s for writing\n", argv[1]);
        return 1;
    }

    userkey = (int16_t *)malloc(sizeof(int16_t) * SECRET_KEY_LENGTH);

    if (userkey == NULL)
    {
        fprintf(stderr, "Error allocating userkey\n");
        return 1;
    }

    srand(3899);

    for (i = 0; i < SECRET_KEY_LENGTH; i++)
    {
        userkey[i] = (int16_t)rand();
    }

    if (fwrite(userkey, sizeof(int16_t), SECRET_KEY_LENGTH, out) !=
            SECRET_KEY_LENGTH)
    {
        fprintf(stderr, "Error writing user key to %s\n", argv[1]);
        return 1;
    }

    return 0;
}
