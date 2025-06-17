#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
Try 0
- read : coalesced
- write : not coalesced
*/
__global__ void tranposeKernel_1D_0(float *A_d, float *B_d, long long n)
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / n;
    int col = tid % n;

    if (row < n && col < n)
    {
        B_d[col * n + row] = A_d[row * n + col];
    }
}