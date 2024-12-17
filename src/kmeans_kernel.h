#include <cuda_runtime.h>
#include <stdio.h>
#include "argparse.h"
#include "helpers.h"
#include "sequentialKmeans.h"

#define CHECK(call)                                    \
{                                                     \
    const cudaError_t error = call;                  \
    if (error != cudaSuccess) {                       \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10 * error);                            \
    }                                                 \
}


int kmeans_cuda(options_t* opts);
int kmeans_cuda_shared(options_t * opts);

__global__ void mapData(double* input, double* oldCentroids,double* centroids, double* labels,int* no_c, int size, int dims, int k);

__global__ void avgData(double* centroids, double* labels, int dims, int k);

__global__ void checkConvergence(double* oldCentroids, double* centroids, int k, int dims, double threshhold, int* converged);

__global__ void mapDataShared(double* input, double* centroids, double* labels, int* no_c, int size, int dims, int k);

//__global__ void avgDataShared(double* centroids, double* labels, int dims, int k);

//__global__ void checkConvergenceShared(double* oldCentroids, double* centroids, int k, int dims, double threshhold, int* converged);
