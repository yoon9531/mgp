#include <cudaruntime.h>
#include <device_launch_parameters.h>

const int SKEW = 1;
/*
Try 4
- Corner turning
- Using shared memory
- Use 32x32 blocks (for coalescing)
- Use padding to avoid bank conflicts
*/
__global__ void transposeKernel_2D_SM_skew(float *A_d, float *B_d, long long n)
{
    extern __shared__ float buffer[];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    const int tileWidth = blockDim.x; // thread 개수 == tile width
    const int SMWidth = blockDim.x + SKEW; // Shared memory width with padding
    
    int srcRow = blockRow * tileWidth + localRow;
    int srcCol = blockCol * tileWidth + localCol;

    // Coalesced read
    // Load data into shared memory
    if (srcRow < n && srcCol < n)
    {
        buffer[localRow * SMWidth + localCol] = A_d[srcRow * n + srcCol];
    }

    __syncthreads(); // Ensure all threads have loaded their data

    // Coalesced write
    int dstRow = blockCol * tileWidth + localRow;
    int dstCol = blockRow * tileWidth + localCol;

    if (dstRow < n && dstCol < n)
    {
        B_d[dstRow * n + dstCol] = buffer[localCol * SMWidth + localRow]; // NO bank conflict
    }
}