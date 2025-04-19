#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

// Kernel function to be executed on the GPU
// This function will be called from the CPU(Host) and executed on the GPU(Device)
// The __global__ keyword indicates that this function is a kernel that runs on the GPU
__global__ void hello_cuda() {
    printf("Hello from CUDA thread %d\n", threadIdx.x);
}


int main(void) {
    printf("Hello GPU from CPU\n");
    hello_cuda<<<1, 10>>>(); // Launch kernel with 10 threads

    // Synchronize to ensure all threads complete before exiting
    cudaError_t err = cudaDeviceSynchronize();

    // Check for errors in kernel launch (e.g., out of memory)
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}