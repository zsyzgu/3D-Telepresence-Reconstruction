#include "CudaHandleError.h"
#include "Parameters.h"
#include "DepthFilter.h"
#include "Timer.h"

__global__ void kernelSFVertical(UINT16* source, float* result)
{
	const int RADIUS = 5;
	const float ALPHA = 0.5f;
	const float THRESHOLD = 20.0f;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < DEPTH_W && y < DEPTH_H) {
		int id = y * DEPTH_W + x;

		float center = source[id];
		if (center == 0) {
			result[id] = 0;
		} else {
			float sum = center;
			float weight = 1;
			float w = 1;
			for (int r = 1; r <= RADIUS; r++) {
				w *= ALPHA;
				if (id - r * DEPTH_W >= 0 && source[id - r * DEPTH_W] != 0 && fabs(source[id - r * DEPTH_W] - center) < THRESHOLD) {
					weight += w;
					sum += w * source[id - r * DEPTH_W];
				}
				if (id + r * DEPTH_W < DEPTH_H * DEPTH_W && source[id + r * DEPTH_W] != 0 && fabs(source[id + r * DEPTH_W] - center) < THRESHOLD) {
					weight += w;
					sum += w * source[id + r * DEPTH_W];
				}
			}
			result[id] = sum / weight;
		}
	}
}

__global__ void kernelSFHorizontal(float* source, float* result)
{
	const int RADIUS = 5;
	const float ALPHA = 0.5f;
	const float THRESHOLD = 20.0f;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < DEPTH_W && y < DEPTH_H) {
		int id = y * DEPTH_W + x;

		float center = source[id];
		if (center == 0) {
			result[id] = 0;
		}
		else {
			float sum = center;
			float weight = 1;
			float w = 1;
			for (int r = 1; r <= RADIUS; r++) {
				w *= ALPHA;
				if (id - r >= 0 && source[id - r] != 0 && fabs(source[id - r] - center) < THRESHOLD) {
					weight += w;
					sum += w * source[id - r];
				}
				if (id + r < DEPTH_H * DEPTH_W && source[id + r] != 0 && fabs(source[id + r] - center) < THRESHOLD) {
					weight += w;
					sum += w * source[id + r];
				}
			}
			result[id] = sum / weight;
		}
	}
}

extern "C"
void cudaDepthFiltering(UINT16* depthMap) {
	UINT16* depth_device;
	HANDLE_ERROR(cudaMalloc(&depth_device, DEPTH_H * DEPTH_W * sizeof(UINT16)));
	HANDLE_ERROR(cudaMemcpy(depth_device, depthMap, DEPTH_H * DEPTH_W * sizeof(UINT16), cudaMemcpyHostToDevice));

	float* depthFloat_device;
	HANDLE_ERROR(cudaMalloc(&depthFloat_device, DEPTH_H * DEPTH_W * sizeof(float)));

	float* depthFloat_device2;
	HANDLE_ERROR(cudaMalloc(&depthFloat_device2, DEPTH_H * DEPTH_W * sizeof(float)));

	dim3 threadsPerBlock = dim3(16, 16);
	dim3 blocksPerGrid = dim3((DEPTH_W + threadsPerBlock.x - 1) / threadsPerBlock.x, (DEPTH_H + threadsPerBlock.y - 1) / threadsPerBlock.y);

	kernelSFVertical << <blocksPerGrid, threadsPerBlock >> > (depth_device, depthFloat_device);
	cudaGetLastError();
	kernelSFHorizontal << <blocksPerGrid, threadsPerBlock >> > (depthFloat_device, depthFloat_device2);
	cudaGetLastError();

	float* depthFloat = new float[DEPTH_H * DEPTH_W];
	HANDLE_ERROR(cudaMemcpy(depthFloat, depthFloat_device2, DEPTH_H * DEPTH_W * sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < DEPTH_H * DEPTH_W; i++) {
		depthMap[i] = depthFloat[i];
	}

	HANDLE_ERROR(cudaFree(depth_device));
	HANDLE_ERROR(cudaFree(depthFloat_device));
	HANDLE_ERROR(cudaFree(depthFloat_device2));
	delete[] depthFloat;
}
