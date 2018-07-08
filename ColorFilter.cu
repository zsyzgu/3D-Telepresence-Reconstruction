#include "CudaHandleError.h"
#include "Parameters.h"
#include "ColorFilter.h"
#include "Timer.h"

__global__ void kernelColorFiltering(UINT8* data, uchar4* color) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < COLOR_W && y < COLOR_H) {
		int id = y * COLOR_W + x;
		INT16 Y = data[id * 2];
		INT16 U = data[(id - (id & 1)) * 2 + 1];
		INT16 V = data[(id - (id & 1)) * 2 + 3];
		INT16 C = Y - 16;
		INT16 D = U - 128;
		INT16 E = V - 128;
		color[id] = make_uchar4(
			max(0, min(255, (298 * C + 409 * E + 128) >> 8)),
			max(0, min(255, (298 * C - 100 * D - 208 * E + 128) >> 8)),
			max(0, min(255, (298 * C + 516 * D + 128) >> 8)),
			0);
	}
}

extern "C"
void cudaColorFilterInit(UINT8*& data_device, RGBQUAD*& color_device) {
	HANDLE_ERROR(cudaMalloc(&data_device, 2 * COLOR_H * COLOR_W * sizeof(UINT8)));
	HANDLE_ERROR(cudaMalloc(&color_device, MAX_CAMERAS * COLOR_H * COLOR_W * sizeof(RGBQUAD)));
}
extern "C"
void cudaColorFilterClean(UINT8*& data_device, RGBQUAD*& color_device) {
	HANDLE_ERROR(cudaFree(data_device));
	HANDLE_ERROR(cudaFree(color_device));
}

extern "C"
void cudaColorFiltering(UINT8* colorMap, UINT8* data_device, RGBQUAD* color_device)
{
	dim3 threadsPerBlock = dim3(256, 1);
	dim3 blocksPerGrid = dim3((COLOR_W + threadsPerBlock.x - 1) / threadsPerBlock.x, (COLOR_H + threadsPerBlock.y - 1) / threadsPerBlock.y);
	
	HANDLE_ERROR(cudaMemcpy(data_device, colorMap, 2 * COLOR_H * COLOR_W * sizeof(UINT8), cudaMemcpyHostToDevice));

	kernelColorFiltering << <blocksPerGrid, threadsPerBlock >> > (data_device, (uchar4*)color_device);
	cudaGetLastError();
}
