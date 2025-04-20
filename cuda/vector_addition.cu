#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <chrono>

#define DATA_NUM 65536// Size of vectors

using namespace std;

__global__ void vector_addition_kernel(float *A_d, float *B_d, float *C_d, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread index
    if (i < DATA_NUM) {
        C_d[i] = A_d[i] + B_d[i]; // Perform vector addition
    }
}


int main() {
    float *A_h, *B_h, *C_h, *C; // Host pointers
    float *A_d, *B_d, *C_d; // Device pointers

    int memsize = DATA_NUM * sizeof(float); // Size of vectors in bytes
    printf("%d elements, memsize = %d bytes\n", DATA_NUM, memsize);

    // Allocate memory on the host
    A_h = (float *)malloc(memsize);
    memset(A_h, 0, memsize); 
    B_h = (float *)malloc(memsize);
    memset(B_h, 0, memsize);
    C_h = (float *)malloc(memsize);
    memset(C_h, 0, memsize);
    C = (float *)malloc(memsize);
    memset(C, 0, memsize);

    // Initialize data
    for(int i = 0; i < DATA_NUM; i++){
        A_h[i] = rand() % 10; // Random values between 0 and 9
        B_h[i] = rand() % 10; 
    }

    // Measure time for CPU computation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < DATA_NUM; i++){
        C[i] = A_h[i] + B_h[i]; // Perform vector addition on host
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
    printf("CPU computation time: %f seconds\n", duration_cpu.count());

    // Allocate memory on the device
    auto start_cudamalloc = std::chrono::high_resolution_clock::now();
    cudaMalloc((void **)&A_d, memsize);
    cudaMemcpy(A_d, A_h, memsize, cudaMemcpyHostToDevice); // Copy data from host to device
    cudaMalloc((void **)&B_d, memsize);
    cudaMemcpy(B_d, B_h, memsize, cudaMemcpyHostToDevice); // Copy data from host to device
    cudaMalloc((void **)&C_d, memsize);
    auto end_cudamalloc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cudamalloc = end_cudamalloc - start_cudamalloc;
    printf("CUDA malloc time: %f seconds\n", duration_cudamalloc.count());


    // Lanch kernel
    auto start_kernel = std::chrono::high_resolution_clock::now();
    vector_addition_kernel<<<ceil(DATA_NUM / 256), 256>>>(A_d, B_d, C_d, DATA_NUM); // Launch kernel with 256 threads per block
    cudaDeviceSynchronize(); // Wait for kernel to finish
    auto end_kernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_kernel = end_kernel - start_kernel;
    printf("Kernel execution time: %f seconds\n", duration_kernel.count());

    // Copy result back to host
    cudaMemcpy(C_h, C_d, memsize, cudaMemcpyDeviceToHost);

    // Release device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    // check result
    bool check = true;
    for(int i = 0; i < DATA_NUM; i++) {
        if (C_h[i] != C[i]) {
            check = false;
            break;
        }
    }
    if (check) {
        printf("Result is correct!\n");
    } else {
        printf("Result is incorrect!\n");
    }

    // Release host memory
    free(A_h);
    free(B_h);
    free(C_h);


    return 0;
}