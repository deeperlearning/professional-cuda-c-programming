#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/**
 * This example implements a very simple two-stage NBody simulation. The goal of
 * this sample code is to illustrate the use of all three main concepts from
 * Chapter 7 in a single application.
 *
 * This NBody simulation consists of two main stages: updating particle
 * velocities based on calculated acceleration, followed by updating particle
 * positions based on the computed velocities.
 *
 * This example also supports the use of compile-time flags -DSINGLE_PREC and
 * -DDOUBLE_PREC to switch between the floating-point types used to store
 * particle acceleration, velocity, and position.
 *
 * Another supported compile-time flag is -DVALIDATE, which turns on executing a
 * copy of the same computation on the host side with the same floating-point
 * type. Using the host values as a baseline, this application can validate its
 * own numerical results. The measure used for validation is the mean distance
 * between a particle's position as calculated on the device versus the position
 * from the host.
 **/

/*
 * If neither single- or double-precision is specified, default
 * to single-precision.
 */
#ifndef SINGLE_PREC
#ifndef DOUBLE_PREC
#define SINGLE_PREC
#endif
#endif

#ifdef SINGLE_PREC

typedef float real;
#define MAX_DIST    200.0f
#define MAX_SPEED   100.0f
#define MASS        2.0f
#define DT          0.00001f
#define LIMIT_DIST  0.000001f
#define POW(x,y)    powf(x,y)
#define SQRT(x)     sqrtf(x)

#else // SINGLE_PREC

typedef double real;
#define MAX_DIST    200.0
#define MAX_SPEED   100.0
#define MASS        2.0
#define DT          0.00001
#define LIMIT_DIST  0.000001
#define POW(x,y)    pow(x,y)
#define SQRT(x)     sqrt(x)

#endif // SINGLE_PREC

#ifdef VALIDATE

/**
 * Host implementation of the NBody simulation.
 **/
static void h_nbody_update_velocity(real *px, real *py,
                                    real *vx, real *vy,
                                    real *ax, real *ay,
                                    int N, int *exceeded_speed, int id)
{
    real total_ax = 0.0f;
    real total_ay = 0.0f;

    real my_x = px[id];
    real my_y = py[id];

    int i = (id + 1) % N;

    while (i != id)
    {
        real other_x = px[i];
        real other_y = py[i];

        real rx = other_x - my_x;
        real ry = other_y - my_y;

        real dist2 = rx * rx + ry * ry;

        if (dist2 < LIMIT_DIST)
        {
            dist2 = LIMIT_DIST;
        }

        real dist6 = dist2 * dist2 * dist2;
        real s = MASS * (1.0f / SQRT(dist6));
        total_ax += rx * s;
        total_ay += ry * s;

        i = (i + 1) % N;
    }

    ax[id] = total_ax;
    ay[id] = total_ay;

    vx[id] = vx[id] + ax[id];
    vy[id] = vy[id] + ay[id];

    real v = SQRT(POW(vx[id], 2.0) + POW(vy[id], 2.0));

    if (v > MAX_SPEED)
    {
        *exceeded_speed = *exceeded_speed + 1;
    }
}

static void h_nbody_update_position(real *px, real *py,
                                    real *vx, real *vy,
                                    int N, int *beyond_bounds, int id)
{

    px[id] += (vx[id] * DT);
    py[id] += (vy[id] * DT);

    real dist = SQRT(POW(px[id], 2.0) + POW(py[id], 2.0));

    if (dist > MAX_DIST)
    {
        *beyond_bounds = 1;
    }
}
#endif // VALIDATE

/**
 * CUDA implementation of simple NBody.
 **/
__global__ void d_nbody_update_velocity(real *px, real *py,
                                        real *vx, real *vy,
                                        real *ax, real *ay,
                                        int N, int *exceeded_speed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    real total_ax = 0.0f;
    real total_ay = 0.0f;

    if (tid >= N) return;

    real my_x = px[tid];
    real my_y = py[tid];

    int i = (tid + 1) % N;

    while (i != tid)
    {
        real other_x = px[i];
        real other_y = py[i];

        real rx = other_x - my_x;
        real ry = other_y - my_y;

        real dist2 = rx * rx + ry * ry;

        if (dist2 < LIMIT_DIST)
        {
            dist2 = LIMIT_DIST;
        }

        real dist6 = dist2 * dist2 * dist2;
        real s = MASS * (1.0f / SQRT(dist6));
        total_ax += rx * s;
        total_ay += ry * s;

        i = (i + 1) % N;
    }

    ax[tid] = total_ax;
    ay[tid] = total_ay;

    vx[tid] = vx[tid] + ax[tid];
    vy[tid] = vy[tid] + ay[tid];

    real v = SQRT(POW(vx[tid], 2.0) + POW(vy[tid], 2.0));

    if (v > MAX_SPEED)
    {
        atomicAdd(exceeded_speed, 1);
    }
}

__global__ void d_nbody_update_position(real *px, real *py,
                                        real *vx, real *vy,
                                        int N, int *beyond_bounds)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    px[tid] += (vx[tid] * DT);
    py[tid] += (vy[tid] * DT);

    real dist = SQRT(POW(px[tid], 2.0) + POW(py[tid], 2.0));

    if (dist > MAX_DIST)
    {
        *beyond_bounds = 1;
    }
}

static void print_points(real *x, real *y, int N)
{
    int i;

    for (i = 0; i < N; i++)
    {
        printf("%.20e %.20e\n", x[i], y[i]);
    }
}

int main(int argc, char **argv)
{
    int i;
    int N = 30720;
    int block = 256;
    int iter, niters = 50;
    real *d_px, *d_py;
    real *d_vx, *d_vy;
    real *d_ax, *d_ay;
    real *h_px, *h_py;
    int *d_exceeded_speed, *d_beyond_bounds;
    int exceeded_speed, beyond_bounds;
#ifdef VALIDATE
    int id;
    real *host_px, *host_py;
    real *host_vx, *host_vy;
    real *host_ax, *host_ay;
    int host_exceeded_speed, host_beyond_bounds;
#endif // VALIDATE

#ifdef SINGLE_PREC
    printf("Using single-precision floating-point values\n");
#else // SINGLE_PREC
    printf("Using double-precision floating-point values\n");
#endif // SINGLE_PREC

#ifdef VALIDATE
    printf("Running host simulation. WARNING, this might take a while.\n");
#endif // VALIDATE

    h_px = (real *)malloc(N * sizeof(real));
    h_py = (real *)malloc(N * sizeof(real));

#ifdef VALIDATE
    host_px = (real *)malloc(N * sizeof(real));
    host_py = (real *)malloc(N * sizeof(real));
    host_vx = (real *)malloc(N * sizeof(real));
    host_vy = (real *)malloc(N * sizeof(real));
    host_ax = (real *)malloc(N * sizeof(real));
    host_ay = (real *)malloc(N * sizeof(real));
#endif // VALIDATE

    for (i = 0; i < N; i++)
    {
        real x = (rand() % 200) - 100;
        real y = (rand() % 200) - 100;

        h_px[i] = x;
        h_py[i] = y;
#ifdef VALIDATE
        host_px[i] = x;
        host_py[i] = y;
#endif // VALIDATE
    }

    CHECK(cudaMalloc((void **)&d_px, N * sizeof(real)));
    CHECK(cudaMalloc((void **)&d_py, N * sizeof(real)));

    CHECK(cudaMalloc((void **)&d_vx, N * sizeof(real)));
    CHECK(cudaMalloc((void **)&d_vy, N * sizeof(real)));

    CHECK(cudaMalloc((void **)&d_ax, N * sizeof(real)));
    CHECK(cudaMalloc((void **)&d_ay, N * sizeof(real)));

    CHECK(cudaMalloc((void **)&d_exceeded_speed, sizeof(int)));
    CHECK(cudaMalloc((void **)&d_beyond_bounds, sizeof(int)));

    CHECK(cudaMemcpy(d_px, h_px, N * sizeof(real), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_py, h_py, N * sizeof(real), cudaMemcpyHostToDevice));

    CHECK(cudaMemset(d_vx, 0x00, N * sizeof(real)));
    CHECK(cudaMemset(d_vy, 0x00, N * sizeof(real)));
#ifdef VALIDATE
    memset(host_vx, 0x00, N * sizeof(real));
    memset(host_vy, 0x00, N * sizeof(real));
#endif // VALIDATE

    CHECK(cudaMemset(d_ax, 0x00, N * sizeof(real)));
    CHECK(cudaMemset(d_ay, 0x00, N * sizeof(real)));
#ifdef VALIDATE
    memset(host_ax, 0x00, N * sizeof(real));
    memset(host_ay, 0x00, N * sizeof(real));
#endif // VALIDATE

    double start = seconds();

    for (iter = 0; iter < niters; iter++)
    {
        CHECK(cudaMemset(d_exceeded_speed, 0x00, sizeof(int)));
        CHECK(cudaMemset(d_beyond_bounds, 0x00, sizeof(int)));

        d_nbody_update_velocity<<<N / block, block>>>(d_px, d_py, d_vx, d_vy,
                d_ax, d_ay, N, d_exceeded_speed);
        d_nbody_update_position<<<N / block, block>>>(d_px, d_py, d_vx, d_vy,
                N, d_beyond_bounds);

    }

    CHECK(cudaDeviceSynchronize());
    double exec_time = seconds() - start;

#ifdef VALIDATE

    for (iter = 0; iter < niters; iter++)
    {
        printf("iter=%d\n", iter);
        host_exceeded_speed = 0;
        host_beyond_bounds = 0;

        #pragma omp parallel for
        for (id = 0; id < N; id++)
        {
            h_nbody_update_velocity(host_px, host_py, host_vx, host_vy,
                                    host_ax, host_ay, N, &host_exceeded_speed,
                                    id);
        }

        #pragma omp parallel for
        for (id = 0; id < N; id++)
        {
            h_nbody_update_position(host_px, host_py, host_vx, host_vy,
                                    N, &host_beyond_bounds, id);
        }
    }

#endif // VALIDATE

    CHECK(cudaMemcpy(&exceeded_speed, d_exceeded_speed, sizeof(int),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&beyond_bounds, d_beyond_bounds, sizeof(int),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_px, d_px, N * sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_py, d_py, N * sizeof(real), cudaMemcpyDeviceToHost));

    print_points(h_px, h_py, 10);
    printf("Any points beyond bounds? %s, # points exceeded velocity %d/%d\n",
           beyond_bounds > 0 ? "true" : "false", exceeded_speed,
           N);
    printf("Total execution time %f s\n", exec_time);

#ifdef VALIDATE
    double error = 0.0;

    for (i = 0; i < N; i++)
    {
        double dist = sqrt(pow(h_px[i] - host_px[i], 2.0) +
                           pow(h_py[i] - host_py[i], 2.0));
        error += dist;
    }

    error /= N;
    printf("Error = %.20e\n", error);
#endif // VALIDATE

    return 0;
}
