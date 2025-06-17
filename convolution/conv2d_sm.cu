#include <cudaruntime.h>
#include <device_launch_parameters.h>

template <int TILE_H, int TILE_W, int K>
__global__ void Conv2d_SharedMemory(
    const float *X, const float *W, float *Y, 
    int N, int M, int C,
    int H, int W, int H_out, int W_out,
    int W_grid
) {    

    extern __shared__ float shmem[];
    float* X_sh = shmem;
    float* W_sh = shmem + (TILE_H + K - 1)*(TILE_W + K - 1);

    int n = blockIdx.x; int m = blockIdx.y;

    int tileRow = blockIdx.z / W_grid; // Tile row index in the output feature map
    int tileCol = blockIdx.z % W_grid; // Tile column index in the output feature map

    int h0 = tileRow * TILE_H; int w0 = tileCol * TILE_W;
    int th = threadIdx.y; int tw = threadIdx.x;

    float acc = 0.0f;

    for(int c = 0; c < C; c++) {
        // Load filter into shared memory
        if (th < K && tw < K) {
            W_sh[th*K + tw] = W[m*C*K*K + c*K*K + th*K + tw];
        }

        __syncthreads();

        // Load input tile into shared memory
        int in_h = h0 + th; int in_w = w0 + tw;
        if (in_h < H && in_w < W) {
            X_sh[th*(TILE_W + K - 1) + tw] = X[n*C*H*W + c*H*W + in_h*W + in_w];
        } else {
            X_sh[th*(TILE_W + K - 1) + tw] = 0.0f; // Zero padding for out-of-bounds
        }

        __syncthreads();

        // Perform convolution
        if (hh < H_out && ww < W_out) {
            for(int kh = 0; kh < K; kh++) {
                for(int kw = 0; kw < K; kw++) {
                    float x_val = X_sh[(th+p)*(TILE_W + K - 1) + (tw+kw)];
                    float w_val = W_sh[kh*K + kw];
                    acc += x_val * w_val;
                }
            }
        }

        __syncthreads();

        // Write the result to global memory
        int hh = h0 + th; int ww = w0 + tw;
        if (hh < H_out && ww < W_out) {
            Y[n*M*H_out*W_out + m*H_out*W_out + hh*W_out + ww] = acc;
        }
    }
}