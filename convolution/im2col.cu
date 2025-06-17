#include <cudaruntime.h>
#include <device_launch_parameters.h>

__global__
void im2col_kernel(int C, int H, int W, int K, 
				   float* X, float* Y, float* X_col) 
{
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int filter_length = K * K;

	int c = blockIdx.z;
	int h_out = blockIdx.y * blockDim.y + threadIdx.y;
	int w_out = blockIdx.x * blockDim.x + threadIdx.x;

	if (c < C && h_out < H_out && w_out < W_out) {
		// im2col 변환한 matrix의 크기 : (H_out*W_out, filter_length*C);
		// 따라서 C_base는 각 column에서 몇 번째 channel에 해당하는 지를 의미한다.
		int C_base = c * filter_length; // X_col의 row index의 base
		int x = h_out*W_out + w_out; // X_col의 column index

		for(int p = 0; p < K; ++p) {
			for(int q = 0; q < K; ++q) {
				int y = C_base + p*K + q;
				// 원본의 X에서의 index 계산
				int x_idx = (c*H*W) + (h_out+p)*W + (w_out+q);
				
				X_col[y*(H_out*W_out) + x] = X[x_idx];
			}
		}
	}
}