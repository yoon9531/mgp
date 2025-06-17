#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
Try 2
- read : partially coalesced
- write : partially coalesced
*/


__global__ transposeKernel_2D_0(float *A_d, float *B_d, long long n)
{

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < n && col < n)
    {
        B_d[col * n + row] = A_d[row * n + col]; // Partially coalesced reads
    }
}

__global__ transposeKernel_2D_1(float *A_d, float *B_d, long long n) {

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < n && col < n)
    {
        B_d[row * n + col] = A_d[col * n + row]; // Partially coalesced writes
    }
}