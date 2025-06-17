#include <cudaruntime.h>
#include <device_launch_parameters.h>

/*
Try 3
- Corner turning
- Using shared memory
- Use 32x32 blocks (for coalescing)
*/
__global__ void transposeKernel_2D_SM(float *A_d, float *B_d, long long n)
{
    extern __shared__ float buffer[];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    const int tileWidth = blockDim.x; // thread 개수 == tile width

    int srcRow = blockRow * tileWidth + localRow;
    int srcCol = blockCol * tileWidth + localCol;

    // Load data into shared memory
    if (srcRow < n && srcCol < n)
    {
        buffer[localRow * tileWidth + localCol] = A_d[srcRow * n + srcCol];
    }

    __syncthreads(); // Ensure all threads have loaded their data

    // Coalesced write
    int dstRow = blockCol * tileWidth + localRow;
    int dstCol = blockRow * tileWidth + localCol;

    if (dstRow < n && dstCol < n)
    {
        B_d[dstRow * n + dstCol] = buffer[localCol * tileWidth + localRow]; // 32-way bank conflict
    }
}

/* Result
    1. The GBps is significantly higher than the previous attempts except for the Second Try.
    2. The kernel execution time is also significantly lower than the previous attempts.
    3. As we introduced shared memroy, and the block size is 32x32, the bank conflicts are occurred.
*/