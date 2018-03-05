#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <Windows.h>

#define BLOCK_SIZE 16

namespace tsdf {
	int resolutionX;
	int resolutionY;
	int resolutionZ;
	float sizeX;
	float sizeY;
	float sizeZ;
	float centerX;
	float centerY;
	float centerZ;

	float* volume_device;
	UINT16* weight_device;
}
using namespace tsdf;

__global__ void kernelClearVolume(float* volume, UINT16* weight, int resolutionX, int resolutionY, int resolutionZ) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < resolutionX && y < resolutionY) {
		for (int z = 0; z < resolutionZ; z++) {
			int id = x + y * resolutionX + z * resolutionX * resolutionY;
			volume[id] = -1;
			weight[id] = 0;
		}
	}
}

__global__ void kernelIntegrateDepth(
	float* volume,
	UINT16* weight,
	int resolutionX,
	int resolutionY,
	int resolutionZ,
	float volumeSizeX,
	float volumeSizeY,
	float volumeSizeZ,
	float centerX,
	float centerY,
	float centerZ,
	UINT16* depthData,
	float* transformation) {

	const int W = 512;
	const int H = 424;
	const float FX = 367.347;
	const float FY = 367.347;
	const float CX = 260.118; //TODO
	const float CY = 208.079;
	const int TRANC_DIST_MM = 2.1 * max(volumeSizeX, max(volumeSizeY, volumeSizeZ));

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
		
	if (x >= resolutionX || y >= resolutionY) {
		return;
	}

	float oriX = (x - 0.5f) * volumeSizeX + centerX;
	float oriY = (y - 0.5f) * volumeSizeY + centerY;

	for (int z = 0; z < resolutionZ; z++) {
		float oriZ = (z - 0.5f) * volumeSizeZ + centerZ;
		float posX = transformation[0 + 0] * oriX + transformation[0 + 1] * oriY + transformation[0 + 2] * oriZ + transformation[12 + 0];
		float posY = transformation[4 + 0] * oriX + transformation[4 + 1] * oriY + transformation[4 + 2] * oriZ + transformation[12 + 1];
		float posZ = transformation[8 + 0] * oriX + transformation[8 + 1] * oriY + transformation[8 + 2] * oriZ + transformation[12 + 2];

		int cooX = int(posX / posZ * FX + CX);
		int cooY = int(posY / posZ * FY + CY);

		if (posZ > 0 && 0 <= cooX && cooX < W && 0 <= cooY && cooY < H) {
			int depth = depthData[cooY * W + cooX];

			if (depth != 0) {
				float sdf = posZ - depth;
				
				if (sdf >= -TRANC_DIST_MM) {
					float tsdf = min(1.0, sdf / TRANC_DIST_MM);
					int id = x + y * resolutionX + z * resolutionX * resolutionY;
					volume[id] = (volume[id] * weight[id] + tsdf) / (weight[id] + 1);
					weight[id]++;
				}
			}
		}
	}
}

extern "C"
void cudaInitVolume(int resolutionX, int resolutionY, int resolutionZ, float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ) {
	tsdf::resolutionX = resolutionX;
	tsdf::resolutionY = resolutionY;
	tsdf::resolutionZ = resolutionZ;
	tsdf::sizeX = sizeX;
	tsdf::sizeY = sizeY;
	tsdf::sizeZ = sizeZ;
	tsdf::centerX = centerX;
	tsdf::centerY = centerY;
	tsdf::centerZ = centerZ;
	cudaMalloc(&volume_device, resolutionX * resolutionY * resolutionZ * sizeof(float));
	cudaMalloc(&weight_device, resolutionX * resolutionY * resolutionZ * sizeof(UINT16));
}

extern "C"
void cudaReleaseVolume() {
	cudaFree(volume_device);
	cudaFree(weight_device);
}

extern "C"
void cudaClearVolume() {
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((resolutionX + BLOCK_SIZE - 1) / BLOCK_SIZE, (resolutionY + BLOCK_SIZE - 1) / BLOCK_SIZE);

	kernelClearVolume << <grid, block >> > (volume_device, weight_device, resolutionX, resolutionY, resolutionZ);
	cudaDeviceSynchronize();
}

extern "C"
void cudaIntegrateDepth(UINT16* depth, float* transformation) {
	float volumeSizeX = sizeX / resolutionX;
	float volumeSizeY = sizeY / resolutionY;
	float volumeSizeZ = sizeZ / resolutionZ;

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((resolutionX + BLOCK_SIZE - 1) / BLOCK_SIZE, (resolutionY + BLOCK_SIZE - 1) / BLOCK_SIZE);

	kernelIntegrateDepth << <grid, block >> > (volume_device, weight_device, resolutionX, resolutionY, resolutionZ, volumeSizeX, volumeSizeY, volumeSizeZ, centerX, centerY, centerZ, depth, transformation);
	cudaDeviceSynchronize();
}

extern "C"
void cudaCalculateMesh(float* result, int& size) {

}
