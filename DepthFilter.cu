#include "CudaHandleError.h"
#include "Parameters.h"
#include "DepthFilter.h"
#include "Timer.h"

namespace FilterNamespace {
	__constant__ int SF_RADIUS = 8;
	__constant__ float SF_ALPHA = 0.65f;
	__constant__ float SF_THRESHOLD = 20.0f;
	__constant__ float TF_ALPHA = 0.5f;
	__constant__ float TF_THRESHOLD = 20.0f;
	float* lastFrame;
};
using namespace FilterNamespace;

__global__ void kernelCleanLastFrame(float* lastFrame) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < DEPTH_W && y < DEPTH_H) {
		int id = y * DEPTH_W + x;
		lastFrame[id] = 0;
	}
}

__global__ void kernelFilterToDisparity(UINT16* source, float* target, float convertFactor) {
	#define DEPTH_SORT(a, b) { if ((a) > (b)) {UINT16 temp = (a); (a) = (b); (b) = temp;} }

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < DEPTH_W && y < DEPTH_H) {
		int id = y * DEPTH_W + x;
		int sid = (y * 2) * (DEPTH_W * 2) + (x * 2);
		UINT16 arr[4] = { source[sid], source[sid + 1], source[sid + DEPTH_W * 2], source[sid + DEPTH_W * 2 + 1] };
		DEPTH_SORT(arr[0], arr[1]);
		DEPTH_SORT(arr[0], arr[2]);
		DEPTH_SORT(arr[0], arr[3]);
		DEPTH_SORT(arr[1], arr[2]);
		DEPTH_SORT(arr[1], arr[3]);
		DEPTH_SORT(arr[2], arr[3]);
		__syncthreads();
		if (arr[0] != 0) {
			target[id] = convertFactor * 2 / (arr[1] + arr[2]);
		} else
		if (arr[1] != 0) {
			target[id] = convertFactor / arr[2];
		} else
		if (arr[2] != 0) {
			target[id] = convertFactor * 2 / (arr[2] + arr[3]);
		} else {
			target[id] = convertFactor / arr[3]; // If arr[3] also equals to zero, just let it be oo.
		}
	}
}

__global__ void kernelFilterToDepth(float* depth, float convertFactor) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < DEPTH_W && y < DEPTH_H) {
		int id = y * DEPTH_W + x;
		depth[id] = convertFactor / depth[id];
	}
}

__global__ void kernelSFVertical(float* depth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < DEPTH_W && y < DEPTH_H) {
		int id = y * DEPTH_W + x;

		float origin = depth[id];
		float result = 0;
		if (origin != 0) {
			float sum = origin;
			float weight = 1;
			float w = 1;
			for (int r = 1; r <= SF_RADIUS; r++) {
				w *= SF_ALPHA;
				if (y - r >= 0 && depth[id - r * DEPTH_W] != 0 && fabs(depth[id - r * DEPTH_W] - origin) <= SF_THRESHOLD) {
					weight += w;
					sum += w * depth[id - r * DEPTH_W];
				}
				if (y + r < DEPTH_H && depth[id + r * DEPTH_W] != 0 && fabs(depth[id + r * DEPTH_W] - origin) <= SF_THRESHOLD) {
					weight += w;
					sum += w * depth[id + r * DEPTH_W];
				}
			}
			result = sum / weight;
		}
		__syncthreads();
		depth[id] = result;
	}
}

__global__ void kernelSFHorizontal(float* depth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < DEPTH_W && y < DEPTH_H) {
		int id = y * DEPTH_W + x;

		float origin = depth[id];
		float result = 0;
		if (origin != 0) {
			float sum = origin;
			float weight = 1;
			float w = 1;
			for (int r = 1; r <= SF_RADIUS; r++) {
				w *= SF_ALPHA;
				if (x - r >= 0 && depth[id - r] != 0 && fabs(depth[id - r] - origin) <= SF_THRESHOLD) {
					weight += w;
					sum += w * depth[id - r];
				}
				if (x + r < DEPTH_W && depth[id + r] != 0 && fabs(depth[id + r] - origin) <= SF_THRESHOLD) {
					weight += w;
					sum += w * depth[id + r];
				}
			}
			result = sum / weight;
		}
		__syncthreads();
		depth[id] = result;
	}
}

__global__ void kernelFillHoles(float* depth) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < DEPTH_W && y < DEPTH_H) {
		int id = y * DEPTH_W + x;
		float result = depth[id];
		if (result == 0) {
			if (x - 1 >= 0) {
				result = max(result, depth[id - 1]);
				if (y - 1 >= 0) result = max(result, depth[id - 1 - DEPTH_W]);
				if (y + 1 < DEPTH_H) result = max(result, depth[id - 1 + DEPTH_W]);
			}
			if (x + 1 < DEPTH_W) {
				result = max(result, depth[id + 1]);
				if (y - 1 >= 0) result = max(result, depth[id + 1 - DEPTH_W]);
				if (y + 1 < DEPTH_H) result = max(result, depth[id + 1 + DEPTH_W]);
			}
			if (y - 1 >= 0) result = max(result, depth[id - DEPTH_W]);
			if (y + 1 < DEPTH_H) result = max(result, depth[id + DEPTH_H]);
		}
		__syncthreads();
		depth[id] = result;
	}
}

__global__ void kernelTemporalFilter(float* depth, float* lastFrame) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < DEPTH_W && y < DEPTH_H) {
		int id = y * DEPTH_W + x;
		float result = depth[id];
		float lastDepth = lastFrame[id];
		if (result == 0) {
			result = lastDepth;
		} else if (lastDepth != 0 && fabs(result - lastDepth) <= TF_THRESHOLD) {
			result = result * TF_ALPHA + lastDepth * (1 - TF_ALPHA);
		}
		__syncthreads();
		depth[id] = result;
		lastFrame[id] = result;
	}
}

extern "C"
void cudaDepthFilterInit(UINT16*& depth_device, float*& depthFloat_device, float*& lastFrame_device) {
	dim3 threadsPerBlock = dim3(256, 1);
	dim3 blocksPerGrid = dim3((DEPTH_W + threadsPerBlock.x - 1) / threadsPerBlock.x, (DEPTH_H + threadsPerBlock.y - 1) / threadsPerBlock.y);

	HANDLE_ERROR(cudaMalloc(&depth_device, (2 * DEPTH_H) * (2 * DEPTH_W) * sizeof(UINT16)));
	HANDLE_ERROR(cudaMalloc(&depthFloat_device, MAX_CAMERAS * DEPTH_H * DEPTH_W * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&lastFrame_device, MAX_CAMERAS * DEPTH_H * DEPTH_W * sizeof(float)));
	for (int i = 0; i < MAX_CAMERAS; i++) {
		kernelCleanLastFrame << <blocksPerGrid, threadsPerBlock >> > (lastFrame_device + i * DEPTH_H * DEPTH_W);
		cudaGetLastError();
	}
}

extern "C"
void cudaDepthFilterClean(UINT16*& depth_device, float*& depthFloat_device, float*& lastFrame_device) {
	HANDLE_ERROR(cudaFree(depth_device));
	HANDLE_ERROR(cudaFree(depthFloat_device));
	HANDLE_ERROR(cudaFree(lastFrame_device));
}

extern "C"
void cudaDepthFiltering(UINT16* depthMap, UINT16* depth_device, float* depthFloat_device, float* lastFrame_device, float convertFactor) {
	dim3 threadsPerBlock = dim3(256, 1);
	dim3 blocksPerGrid = dim3((DEPTH_W + threadsPerBlock.x - 1) / threadsPerBlock.x, (DEPTH_H + threadsPerBlock.y - 1) / threadsPerBlock.y);

	HANDLE_ERROR(cudaMemcpy(depth_device, depthMap, (2 * DEPTH_H) * (2 * DEPTH_W) * sizeof(UINT16), cudaMemcpyHostToDevice));
	kernelFilterToDisparity << <blocksPerGrid, threadsPerBlock >> > (depth_device, depthFloat_device, convertFactor);
	cudaGetLastError();
	kernelSFVertical << <blocksPerGrid, threadsPerBlock >> > (depthFloat_device);
	cudaGetLastError();
	kernelSFHorizontal << <blocksPerGrid, threadsPerBlock >> > (depthFloat_device);
	cudaGetLastError();
	kernelFillHoles << <blocksPerGrid, threadsPerBlock >> > (depthFloat_device);
	cudaGetLastError();
	kernelTemporalFilter << <blocksPerGrid, threadsPerBlock >> > (depthFloat_device, lastFrame_device);
	cudaGetLastError();
	kernelFilterToDepth << <blocksPerGrid, threadsPerBlock >> > (depthFloat_device, convertFactor);
	cudaGetLastError();
}
