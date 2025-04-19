#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

/* Kernel using static shared memory 
    This kernel uses static shared memory to store data that is shared among threads in a block
    The __shared__ keyword indicates that this memory is shared among threads in the same block
    The size of shared memory is defined at compile time
*/
__global__ void staticSharedKernel(int *out) {
    // 16 ints each block
    __shared__ int sdata[16];

    int tid = threadIdx.x;

    //each thread writes tid*2 into shared
    sdata[tid] = tid * 2;
    __syncthreads(); // Synchronize threads in the block

    // copy back to global memory
    out[tid] = sdata[tid];
}

/*
    Kernel using dynamic shared memory 
    This kernel uses dynamic shared memory to store data that is shared among threads in a block
    The size of shared memory is defined at runtime and can be specified when launching the kernel
*/
__global__ void dynamicSharedKernel(int *out) {
    // Size is given by the third launch parameter
    extern __shared__ int sdata[]; // Declare dynamic shared memory

    int tid = threadIdx.x;
    // each thread writes tid*3 into shared
    sdata[tid] = tid * 3;
    __syncthreads(); // Synchronize threads in the block

    // copy back to global memory
    out[tid] = sdata[tid];
}


// Host code
int main() {
    constexpr int blockSize = 16;
    constexpr int nBytes = blockSize * sizeof(int);

    // Allocate output buffer on the host
    int h_out[blockSize];
    memset(h_out, 0, nBytes); // Initialize output buffer

    // Allocate device buffer
    int *devOut;
    cudaMalloc(&devOut, nBytes);

    // 1. Static shared memory kernel
    staticSharedKernel<<<1, blockSize>>>(devOut);
    cudaDeviceSynchronize(); // Wait for kernel to finish

    cudaMemcpy(h_out, devOut, nBytes, cudaMemcpyDeviceToHost); // Copy result back to host
    printf("Static shared memory kernel output:\n");
    for (int i = 0; i < blockSize; i++) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // 2. Dynamic shared memory kernel
    // Launch with third parameter = shared bytes = blockSize * sizeof(int)
    dynamicSharedKernel<<<1, blockSize, nBytes>>>(devOut);
    cudaDeviceSynchronize(); // Wait for kernel to finish

    cudaMemcpy(h_out, devOut, nBytes, cudaMemcpyDeviceToHost); // Copy result back to host
    printf("Dynamic shared memory kernel output:\n");
    for (int i = 0; i < blockSize; i++) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    cudaFree(devOut); // Free device memory

    return 0;
}