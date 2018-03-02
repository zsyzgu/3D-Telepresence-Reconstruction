#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <Windows.h>
#include <vector>

#define BLOCK_SIZE 16

namespace tsdf {
	int resolutionX;
	int resolutionY;
	int resolutionZ;
	float sizeX;
	float sizeY;
	float sizeZ;
	std::vector<float> transformation;

	float* volume_device;
	UINT16* weight_device;

	__global__ void kernelClearVolume(float* volume, UINT16* weight, int resolutionX, int resolutionY, int resolutionZ) {
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < resolutionX && y < resolutionY) {
			for (int z = 0; z < resolutionZ; z++) {
				int id = x + y * resolutionX + z * resolutionX * resolutionY;
				volume[id] = 0;
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
		UINT16* depth,
		std::vector<float> transformation) {
		
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		
		if (x >= resolutionX || y >= resolutionY) {
			return;
		}

		//TODO
	}

	extern "C"
	void cudaInitVolume(int resolutionX, int resolutionY, int resolutionZ, float sizeX, float sizeY, float sizeZ, std::vector<float> transformation) {
		tsdf::resolutionX = resolutionX;
		tsdf::resolutionY = resolutionY;
		tsdf::resolutionZ = resolutionZ;
		tsdf::sizeX = sizeX;
		tsdf::sizeY = sizeY;
		tsdf::sizeZ = sizeZ;
		tsdf::transformation = transformation;
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
	}

	extern "C"
	void cudaIntegrateDepth(UINT16* depth, float* transformation) {
		float volumeSizeX = sizeX / resolutionX;
		float volumeSizeY = sizeY / resolutionY;
		float volumeSizeZ = sizeZ / resolutionZ;

		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((resolutionX + BLOCK_SIZE - 1) / BLOCK_SIZE, (resolutionY + BLOCK_SIZE - 1) / BLOCK_SIZE);

		kernelIntegrateDepth << <grid, block >> > (volume_device, weight_device, resolutionX, resolutionY, resolutionZ, volumeSizeX, volumeSizeY, volumeSizeZ, depth, transformation);
	}

	extern "C"
	void cudaCalculateMesh(std::vector<float>& result) {
		result.resize(0);
	}
}
