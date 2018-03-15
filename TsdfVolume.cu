#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <Windows.h>
#include <iostream>
#include "Timer.h"

#define BLOCK_SIZE 16

namespace tsdf {
	const int W = 512;
	const int H = 424;
	
	int3 resolution;
	float3 size;
	float3 center;

	float* volume_device;
	UINT8* volume_color_device;
	UINT16* depth_device;
	UINT8* color_device;
	float* transformation_device;
	int* count_device;
	int* count_host;

	dim3 grid;
	dim3 block;
}
using namespace tsdf;

__device__ __forceinline__ int devicePid(int x, int y, int3 resolution) {
	int gx = gridDim.x, bx = x / BLOCK_SIZE, tx = x % BLOCK_SIZE;
	int gy = gridDim.y, by = y / BLOCK_SIZE, ty = y % BLOCK_SIZE;
	return (by * gx + bx) * BLOCK_SIZE * BLOCK_SIZE + (ty * BLOCK_SIZE + tx);
}

__device__ __forceinline__ int deviceVid(int x, int y, int z, int3 resolution) {
	return devicePid(x, y, resolution) + z * resolution.x * resolution.y;
}

extern "C"
void cudaInitVolume(int resolutionX, int resolutionY, int resolutionZ, float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ) {
	resolution.x = resolutionX;
	resolution.y = resolutionY;
	resolution.z = resolutionZ;
	size.x = sizeX;
	size.y = sizeY;
	size.z = sizeZ;
	center.x = centerX;
	center.y = centerY;
	center.z = centerZ;
	cudaMalloc(&volume_device, resolution.x * resolution.y * resolution.z * sizeof(float));
	cudaMalloc(&volume_color_device, resolution.x * resolution.y * resolution.z * 4 * sizeof(UINT8));
	cudaMalloc(&depth_device, H * W * sizeof(float));
	cudaMalloc(&color_device, H * W * 4 * sizeof(UINT8));
	cudaMalloc(&transformation_device, 16 * sizeof(float));
	cudaMalloc(&count_device, resolution.x * resolution.y * sizeof(int));
	count_host = new int[resolution.x * resolution.y];
	block = dim3(BLOCK_SIZE, BLOCK_SIZE);
	grid = dim3((resolution.x + BLOCK_SIZE - 1) / BLOCK_SIZE, (resolution.y + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

extern "C"
void cudaReleaseVolume() {
	cudaFree(volume_device);
	cudaFree(depth_device);
	cudaFree(color_device);
	cudaFree(transformation_device);
	cudaFree(count_device);
	delete[] count_host;
}

__global__ void kernelIntegrateDepth(float* volume, UINT8* volume_color, UINT16* depthData, UINT8* colorData, float* transformation, int3 resolution, float3 size, float3 center) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	__shared__ float trans_shared[16];
	if (threadIdx.y == 0) {
		trans_shared[threadIdx.x] = transformation[threadIdx.x];
	}

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	float volumeSizeX = size.x / resolution.x;
	float volumeSizeY = size.y / resolution.y;
	float volumeSizeZ = size.z / resolution.z;
	float offsetX = center.x - size.x / 2;
	float offsetY = center.y - size.y / 2;
	float offsetZ = center.z - size.z / 2;
	const int resolutionXY = resolution.x * resolution.y;

	const int W = 512;
	const int H = 424;
	const float FX = 367.347;
	const float FY = -367.347;
	const float CX = 260.118;
	const float CY = 208.079;
	const float TRANC_DIST_M = 3.1 * max(volumeSizeX, max(volumeSizeY, volumeSizeZ));

	float oriX = (x + 0.5) * volumeSizeX + offsetX;
	float oriY = (y + 0.5) * volumeSizeY + offsetY;
	for (int z = 0; z < resolution.z; z++) {
		float oriZ = (z + 0.5) * volumeSizeZ + offsetZ;
		float posX = trans_shared[0 + 0] * oriX + trans_shared[0 + 1] * oriY + trans_shared[0 + 2] * oriZ + trans_shared[12 + 0];
		float posY = trans_shared[4 + 0] * oriX + trans_shared[4 + 1] * oriY + trans_shared[4 + 2] * oriZ + trans_shared[12 + 1];
		float posZ = trans_shared[8 + 0] * oriX + trans_shared[8 + 1] * oriY + trans_shared[8 + 2] * oriZ + trans_shared[12 + 2];

		int cooX = posX * FX / posZ + CX;
		int cooY = posY * FY / posZ + CY;

		float tsdf = -1;
		uchar3 color;
		color.x = color.y = color.z = 0;
		if (posZ > 0 && 0 <= cooX && cooX < W && 0 <= cooY && cooY < H) {
			UINT16 depth = depthData[cooY * W + cooX];
			
			if (depth != 0) {
				float xl = (cooX - CX) / FX;
				float yl = (cooY - CY) / FY;
				float sdf = depth * 0.001 - rsqrtf((xl * xl + yl * yl + 1) / (posX * posX + posY * posY + posZ * posZ));

				if (sdf >= -TRANC_DIST_M) {
					tsdf = sdf / TRANC_DIST_M;

					if (tsdf < 1.0) {
						color.x = colorData[((cooY * W + cooX) << 2) + 2];
						color.y = colorData[((cooY * W + cooX) << 2) + 1];
						color.z = colorData[((cooY * W + cooX) << 2) + 0];
					} else {
						tsdf = 1.0;
					}
				}
			}
		}
		__syncthreads();
		int id = deviceVid(x, y, z, resolution);
		volume[id] = tsdf;
		__syncthreads();
		volume_color[(id << 2) + 0] = color.x;
		volume_color[(id << 2) + 1] = color.y;
		volume_color[(id << 2) + 2] = color.z;
		__syncthreads();
		
	}
}

extern "C"
void cudaIntegrateDepth(UINT16* depth, RGBQUAD* color, float* transformation) {
	cudaMemcpy(depth_device, depth, H * W * sizeof(UINT16), cudaMemcpyHostToDevice);
	cudaMemcpy(color_device, color, H * W * 4 * sizeof(UINT8), cudaMemcpyHostToDevice);
	cudaMemcpy(transformation_device, transformation, 16 * sizeof(float), cudaMemcpyHostToDevice);

	kernelIntegrateDepth << <grid, block >> > (volume_device, volume_color_device, depth_device, color_device, transformation_device, resolution, size, center);
}

__constant__ UINT8 triNumber_device[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 5, 5, 2, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 2, 3, 3, 4, 3, 4, 2, 3, 3, 4, 4, 5, 4, 5, 3, 2, 3, 4, 4, 3, 4, 5, 3, 2, 4, 5, 5, 4, 5, 2, 4, 1, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3, 2, 3, 3, 4, 3, 4, 4, 5, 3, 2, 4, 3, 4, 3, 5, 2, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 4, 3, 4, 4, 3, 4, 5, 5, 4, 4, 3, 5, 2, 5, 4, 2, 1, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 2, 3, 3, 2, 3, 4, 4, 5, 4, 5, 5, 2, 4, 3, 5, 4, 3, 2, 4, 1, 3, 4, 4, 5, 4, 5, 3, 4, 4, 5, 5, 2, 3, 4, 2, 1, 2, 3, 3, 2, 3, 4, 2, 1, 3, 2, 4, 1, 2, 1, 1, 0};
__constant__ INT8 triTable_device[256][16] =
{ { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
{ 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
{ 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
{ 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1 },
{ 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1 },
{ 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
{ 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
{ 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
{ 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1 },
{ 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
{ 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1 },
{ 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 },
{ 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1 },
{ 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
{ 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1 },
{ 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1 },
{ 8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1 },
{ 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
{ 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
{ 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1 },
{ 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1 },
{ 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1 },
{ 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1 },
{ 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
{ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1 },
{ 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
{ 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1 },
{ 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
{ 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1 },
{ 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
{ 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
{ 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
{ 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1 },
{ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1 },
{ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1 },
{ 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
{ 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
{ 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1 },
{ 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1 },
{ 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
{ 5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1 },
{ 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1 },
{ 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1 },
{ 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1 },
{ 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
{ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
{ 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1 },
{ 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
{ 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1 },
{ 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
{ 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1 },
{ 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1 },
{ 8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1 },
{ 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
{ 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1 },
{ 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1 },
{ 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
{ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1 },
{ 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
{ 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
{ 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1 },
{ 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1 },
{ 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1 },
{ 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1 },
{ 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1 },
{ 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
{ 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
{ 2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
{ 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1 },
{ 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
{ 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1 },
{ 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
{ 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1 },
{ 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1 },
{ 7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
{ 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
{ 8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1 },
{ 6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1 },
{ 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1 },
{ 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1 },
{ 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1 },
{ 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
{ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1 },
{ 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
{ 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1 },
{ 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
{ 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
{ 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1 },
{ 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
{ 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1 },
{ 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1 },
{ 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1 },
{ 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1 },
{ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1 },
{ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1 },
{ 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1 },
{ 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1 },
{ 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1 },
{ 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1 },
{ 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1 },
{ 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1 },
{ 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1 },
{ 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1 },
{ 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1 },
{ 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1 },
{ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1 },
{ 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1 },
{ 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1 },
{ 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1 },
{ 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1 },
{ 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1 },
{ 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1 },
{ 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1 },
{ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1 },
{ 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1 },
{ 2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
{ 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1 },
{ 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1 },
{ 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1 },
{ 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
{ 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1 },
{ 0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1 },
{ 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1 },
{ 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1 },
{ 0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1 },
{ 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1 },
{ 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1 },
{ 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1 },
{ 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1 },
{ 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1 },
{ 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1 },
{ 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1 },
{ 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1 },
{ 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1 },
{ 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1 },
{ 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
{ 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1 },
{ 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1 },
{ 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1 },
{ 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1 },
{ 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
{ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1 },
{ 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1 },
{ 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
{ 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1 },
{ 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 } };

__device__ __forceinline__ UINT16 deviceGetCubeIndex(float* volume, int x, int y, int z, int3 resolution) {
	if (volume[deviceVid(x + 0, y + 0, z + 0, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 1, y + 0, z + 0, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 0, y + 1, z + 0, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 1, y + 1, z + 0, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 0, y + 0, z + 1, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 1, y + 0, z + 1, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 0, y + 1, z + 1, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 1, y + 1, z + 1, resolution)] == -1) return 0;
	int index = 0;
	if (volume[deviceVid(x + 0, y + 0, z + 0, resolution)] < 0) index |= 1;
	if (volume[deviceVid(x + 1, y + 0, z + 0, resolution)] < 0) index |= 2;
	if (volume[deviceVid(x + 0, y + 1, z + 0, resolution)] < 0) index |= 8;
	if (volume[deviceVid(x + 1, y + 1, z + 0, resolution)] < 0) index |= 4;
	if (volume[deviceVid(x + 0, y + 0, z + 1, resolution)] < 0) index |= 16;
	if (volume[deviceVid(x + 1, y + 0, z + 1, resolution)] < 0) index |= 32;
	if (volume[deviceVid(x + 0, y + 1, z + 1, resolution)] < 0) index |= 128;
	if (volume[deviceVid(x + 1, y + 1, z + 1, resolution)] < 0) index |= 64;
	return index;
}

__global__ void kernelMarchingCubesCount(float* volume, int* count, int3 resolution) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x + 1 >= resolution.x || y + 1 >= resolution.y) {
		if (x == resolution.x - 1 || y == resolution.y - 1) {
			count[devicePid(x, y, resolution)] = 0;
		}
		return;
	}

	int cnt = 0;
	for (int z = 0; z + 1 < resolution.z; z++) {
		int index = deviceGetCubeIndex(volume, x, y, z, resolution);
		cnt += triNumber_device[index];
	}

	count[devicePid(x, y, resolution)] = cnt;
}

__device__ __forceinline__ void deviceCalnEdgePoint(float* volume, UINT8* volume_color, int x1, int y1, int z1, int x2, int y2, int z2, float3& pos, uchar3& color, int3 resolution, float3 size, float3 center) {
	const int resolutionXY = resolution.x * resolution.y;
	int id1 = deviceVid(x1, y1, z1, resolution);
	int id2 = deviceVid(x2, y2, z2, resolution);
	float v1 = volume[id1];
	float v2 = volume[id2];
	if ((v1 < 0) ^ (v2 < 0)) {
		float k =  v1 / (v1 - v2);
		float volumeSizeX = size.x / resolution.x;
		float volumeSizeY = size.y / resolution.y;
		float volumeSizeZ = size.z / resolution.z;
		float offsetX = center.x - size.x / 2;
		float offsetY = center.y - size.y / 2;
		float offsetZ = center.z - size.z / 2;
		pos.x = ((1 - k) * x1 + k * x2 - 0.5) * volumeSizeX + offsetX;
		pos.y = ((1 - k) * y1 + k * y2 - 0.5) * volumeSizeY + offsetY;
		pos.z = ((1 - k) * z1 + k * z2 - 0.5) * volumeSizeZ + offsetZ;
		color.x = (UINT8)min((1 - k) * volume_color[(id1 << 2) + 0] + k * volume_color[(id2 << 2) + 0], 255.0);
		color.y = (UINT8)min((1 - k) * volume_color[(id1 << 2) + 1] + k * volume_color[(id2 << 2) + 1], 255.0);
		color.z = (UINT8)min((1 - k) * volume_color[(id1 << 2) + 2] + k * volume_color[(id2 << 2) + 2], 255.0);
	}
}

__global__ void kernelMarchingCubes(float* volume, UINT8* volume_color, int* count, float* tris, UINT8* tris_color, int3 resolution, float3 size, float3 center) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x + 1 >= resolution.x || y + 1 >= resolution.y) {
		return;
	}

	float3 pos[12];
	uchar3 color[12];

	int tid = count[devicePid(x, y, resolution)];
	for (int z = 0; z + 1 < resolution.z; z++) {
		int index = deviceGetCubeIndex(volume, x, y, z, resolution);

		deviceCalnEdgePoint(volume, volume_color, x + 0, y + 0, z + 0, x + 1, y + 0, z + 0, pos[0], color[0], resolution, size, center);
		deviceCalnEdgePoint(volume, volume_color, x + 1, y + 0, z + 0, x + 1, y + 1, z + 0, pos[1], color[1], resolution, size, center);
		deviceCalnEdgePoint(volume, volume_color, x + 1, y + 1, z + 0, x + 0, y + 1, z + 0, pos[2], color[2], resolution, size, center);
		deviceCalnEdgePoint(volume, volume_color, x + 0, y + 1, z + 0, x + 0, y + 0, z + 0, pos[3], color[3], resolution, size, center);
		
		deviceCalnEdgePoint(volume, volume_color, x + 0, y + 0, z + 1, x + 1, y + 0, z + 1, pos[4], color[4], resolution, size, center);
		deviceCalnEdgePoint(volume, volume_color, x + 1, y + 0, z + 1, x + 1, y + 1, z + 1, pos[5], color[5], resolution, size, center);
		deviceCalnEdgePoint(volume, volume_color, x + 1, y + 1, z + 1, x + 0, y + 1, z + 1, pos[6], color[6], resolution, size, center);
		deviceCalnEdgePoint(volume, volume_color, x + 0, y + 1, z + 1, x + 0, y + 0, z + 1, pos[7], color[7], resolution, size, center);

		deviceCalnEdgePoint(volume, volume_color, x + 0, y + 0, z + 0, x + 0, y + 0, z + 1, pos[8], color[8], resolution, size, center);
		deviceCalnEdgePoint(volume, volume_color, x + 1, y + 0, z + 0, x + 1, y + 0, z + 1, pos[9], color[9], resolution, size, center);
		deviceCalnEdgePoint(volume, volume_color, x + 1, y + 1, z + 0, x + 1, y + 1, z + 1, pos[10], color[10], resolution, size, center);
		deviceCalnEdgePoint(volume, volume_color, x + 0, y + 1, z + 0, x + 0, y + 1, z + 1, pos[11], color[11], resolution, size, center);

		for (int i = 0; i < 5; i++) {
			if (triTable_device[index][i * 3] != -1) {
				for (int j = 0; j < 3; j++) {
					int edgeId = triTable_device[index][i * 3 + j];
					tris[tid * 9 + j * 3 + 0] = pos[edgeId].x;
					tris[tid * 9 + j * 3 + 1] = pos[edgeId].y;
					tris[tid * 9 + j * 3 + 2] = pos[edgeId].z;
					tris_color[tid * 9 + j * 3 + 0] = color[edgeId].x;
					tris_color[tid * 9 + j * 3 + 1] = color[edgeId].y;
					tris_color[tid * 9 + j * 3 + 2] = color[edgeId].z;
				}
				tid++;
			} else {
				break;
			}
		}
	}
}

int cudaCountAccumulation() {
	cudaMemcpy(count_host, count_device, resolution.x * resolution.y * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 1; i < resolution.x * resolution.y; i++) {
		count_host[i] += count_host[i - 1];
	}
	for (int i = resolution.x * resolution.y - 1; i >= 1; i--) {
		count_host[i] = count_host[i - 1];
	}
	count_host[0] = 0;
	int tris_size = count_host[resolution.x * resolution.y - 1];
	cudaMemcpy(count_device, count_host, resolution.x * resolution.y * sizeof(int), cudaMemcpyHostToDevice);
	return tris_size;
}

extern "C"
void cudaCalculateMesh(float*& tris, UINT8*& tris_color, int& tri_size) {
	Timer timer;

	kernelMarchingCubesCount << <grid, block >> > (volume_device, count_device, resolution);
	cudaDeviceSynchronize();

	timer.outputTime();

	tri_size = cudaCountAccumulation();

	float* tris_device;
	UINT8* tris_color_device;
	cudaMalloc(&tris_device, tri_size * 9 * sizeof(float));
	cudaMalloc(&tris_color_device, tri_size * 9 * sizeof(UINT8));

	kernelMarchingCubes << <grid, block >> > (volume_device, volume_color_device, count_device, tris_device, tris_color_device, resolution, size, center);

	tris = new float[tri_size * 9];
	tris_color = new UINT8[tri_size * 9];
	cudaMemcpy(tris, tris_device, tri_size * 9 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(tris_color, tris_color_device, tri_size * 9 * sizeof(UINT8), cudaMemcpyDeviceToHost);
	
	cudaFree(tris_device);
	cudaFree(tris_color_device);
}
