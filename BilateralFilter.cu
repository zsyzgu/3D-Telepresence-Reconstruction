#include "CudaHandleError.h"
#include "Parameters.h"
#include <Windows.h>
#include <math.h>

#define BLOCK_SIZE 16

__global__ void kernelBilateralFiltering(UINT16* depth, UINT16* output, int H, int W) {
	const int RADIUS = 9;
	const float SIGMA_D = 3;
	const float SIGMA_I = 10;
	const float SIGMA_D_C = 2 * SIGMA_D * SIGMA_D;
	const float SIGMA_I_C = 2 * SIGMA_I * SIGMA_I;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < W && y < H) {
		float sum = 0;
		float val = 0;

		for (int dx = -RADIUS; dx <= RADIUS; dx++) {
			for (int dy = -RADIUS; dy <= RADIUS; dy++) {
				int nx = x + dx;
				int ny = y + dy;

				if (0 <= nx && nx < W && 0 <= ny && ny < H) {
					int distD2 = (dx * dx + dy * dy);
					int distI2 = (depth[y * W + x] - depth[ny * W + nx]);
					distI2 = distI2 * distI2;
					float v = exp(-distD2 / SIGMA_D_C - distI2 / SIGMA_I_C);
					val += v;
					sum += v * depth[ny * W + nx];
				}
			}
		}

		if (val > 0) {
			output[y * W + x] = (UINT16)(sum / val);
		}
	}
}

extern "C"
void cudaBilateralFiltering(UINT16* depth) {
	static const int H = DEPTH_H;
	static const int W = DEPTH_W;
	static const int n = H * W;

	UINT16* d_depth;
	UINT16* d_output;
	HANDLE_ERROR(cudaMalloc(&d_depth, n * sizeof(UINT16)));
	HANDLE_ERROR(cudaMalloc(&d_output, n * sizeof(UINT16)));

	HANDLE_ERROR(cudaMemcpy(d_depth, depth, n * sizeof(UINT16), cudaMemcpyHostToDevice));

	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blockNum((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernelBilateralFiltering << <blockNum, blockSize >> > (d_depth, d_output, H, W);
	HANDLE_ERROR(cudaGetLastError());

	HANDLE_ERROR(cudaMemcpy(depth, d_output, n * sizeof(UINT16), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(d_depth));
	HANDLE_ERROR(cudaFree(d_output));
}
