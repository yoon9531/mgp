#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#define TILE_WIDTH 2
#define WIDTH 4

__global__ void MatrixMulKernelNaive(float *d_M, float *d_N, float *d_P, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate column index

    float Pvalue = 0; // Initialize Pvalue to 0

    // Perform matrix multiplication
    for (int k = 0; k < N; ++k) {
        Pvalue += d_M[row * N + k] * d_N[k * N + col];
    }

    d_P[row * N + col] = Pvalue; // Store the result in the output matrix
}

__global__ void MatrixMulKernelShared(float *M, float *N, float *P, int width) {
    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH]; // Shared memory for matrix M
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH]; // Shared memory for matrix N

    int bx = blockIdx.x; // Block index in x direction
    int by = blockIdx.y; // Block index in y direction
    int tx = threadIdx.x; // Thread index in x direction
    int ty = threadIdx.y; // Thread index in y direction

    int Row = by * TILE_WIDTH + ty; // Calculate row index
    int Col = bx * TILE_WIDTH + tx; // Calculate column index

    float Pvalue = 0; 

    for(int m = 0; m < WIDTH / TILE_WIDTH; ++m) {
        // Load shared memory phase
        subTileM[ty][tx] = M[Row * WIDTH + (m * TILE_WIDTH + tx)]; // Load M tile into shared memory
        subTileN[ty][tx] = N[(m * TILE_WIDTH + ty) * WIDTH + Col]; // Load N tile into shared memory
        __syncthreads(); // Wait for the copy to finish

        // Compute phase
        for(int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += subTileM[ty][k] * subTileN[k][tx]; // Perform multiplication and accumulate
        }
        __syncthreads(); // Wait for all threads to finish before loading the next tile

    }

    P[Row * WIDTH + Col] = Pvalue; // Store the result in the output matrix

}

int main() {

}