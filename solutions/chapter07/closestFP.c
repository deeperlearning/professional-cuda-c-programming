#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: %s <value>\n", argv[0]);
        return 1;
    }

    double d = atof(argv[1]);
    float f = atof(argv[1]);

    printf("FLT_MAX = %e, DBL_MAX = %e\n", FLT_MAX, DBL_MAX);

    printf("Double-precision: value = %.20f, next highest = %.20f, next lowest "
            "= %.20f\n", d, nextafter(d, d + 1.0), nextafter(d, d - 1.0));
    printf("Single-precision: value = %.20f, next highest = %.20f, next lowest "
            "= %.20f\n", f, nextafterf(f, f + 1.0f), nextafterf(f, f - 1.0f));

    return 0;
}
