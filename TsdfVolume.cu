#include "CudaHandleError.h"
#include "Parameters.h"
#include <Windows.h>
#include <iostream>
#include "Timer.h"
#include "Vertex.h"

#define BLOCK_SIZE 16 // FIXED !!! for optimization
#define BLOCK_LOG 4
#define MAX_CAMERAS 8

namespace tsdf {
	int3 resolution;
	float3 size;
	float3 center;
	float3 volumeSize;
	float3 offset;

	float* volume_device;
	uchar4* volume_color_device;
	UINT8* cubeIndex_device;
	cudaChannelFormatDesc depthDesc;
	cudaChannelFormatDesc colorDesc;
	cudaArray* depth_device;
	cudaArray* color_device;
	texture<UINT16, 2, cudaReadModeElementType> depthTexture;
	texture<uchar4, 2, cudaReadModeElementType> colorTexture;
	float* transformation_device;
	int* count_device;
	int* count_host;

	dim3 grid;
	dim3 block;
}
using namespace tsdf;

__device__ __forceinline__ int devicePid(int x, int y, int3 resolution) {
	int bx = x >> BLOCK_LOG, tx = x ^ (bx << BLOCK_LOG);
	int by = y >> BLOCK_LOG, ty = y ^ (by << BLOCK_LOG);
	return ((((by * gridDim.x + bx) << BLOCK_LOG) + ty) << BLOCK_LOG) + tx;
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
	volumeSize.x = size.x / resolution.x;
	volumeSize.y = size.y / resolution.y;
	volumeSize.z = size.z / resolution.z;
	offset.x = center.x - size.x / 2;
	offset.y = center.y - size.y / 2;
	offset.z = center.z - size.z / 2;
	HANDLE_ERROR(cudaMalloc(&volume_device, resolution.x * resolution.y * resolution.z * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&volume_color_device, resolution.x * resolution.y * resolution.z * sizeof(uchar4)));
	HANDLE_ERROR(cudaMalloc(&cubeIndex_device, resolution.x * resolution.y * resolution.z * sizeof(UINT8)));
	depthDesc = cudaCreateChannelDesc<UINT16>();
	HANDLE_ERROR(cudaMallocArray(&depth_device, &depthDesc, DEPTH_W, DEPTH_H));
	colorDesc = cudaCreateChannelDesc<uchar4>();
	HANDLE_ERROR(cudaMallocArray(&color_device, &colorDesc, COLOR_W, COLOR_H));

	HANDLE_ERROR(cudaMalloc(&transformation_device, 16 * sizeof(float) * MAX_CAMERAS));
	HANDLE_ERROR(cudaMalloc(&count_device, resolution.x * resolution.y * sizeof(int)));
	count_host = new int[resolution.x * resolution.y];
	block = dim3(BLOCK_SIZE, BLOCK_SIZE);
	grid = dim3((resolution.x + BLOCK_SIZE - 1) / BLOCK_SIZE, (resolution.y + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

extern "C"
void cudaReleaseVolume() {
	HANDLE_ERROR(cudaFree(volume_device));
	HANDLE_ERROR(cudaFree(volume_color_device));
	HANDLE_ERROR(cudaFree(cubeIndex_device));
	HANDLE_ERROR(cudaFreeArray(depth_device));
	HANDLE_ERROR(cudaFreeArray(color_device));
	HANDLE_ERROR(cudaFree(transformation_device));
	HANDLE_ERROR(cudaFree(count_device));
	delete[] count_host;
}

__global__ void kernelIntegrateDepth(int cameras, float* volume, uchar4* volume_color, float* transformation, int3 resolution, float3 volumeSize, float3 offset) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	const float TRANC_DIST_M = 3.1 * max(volumeSize.x, max(volumeSize.y, volumeSize.z));
	uchar4 color;
	color.x = color.y = color.z = 255;

	float oriX = (x + 0.5) * volumeSize.x + offset.x;
	float oriY = (y + 0.5) * volumeSize.y + offset.y;
	float oriZ = (0 - 0.5) * volumeSize.z + offset.z;
	float posX = transformation[0] * oriX + transformation[1] * oriY + transformation[2] * oriZ + transformation[12];
	float posY = transformation[4] * oriX + transformation[5] * oriY + transformation[6] * oriZ + transformation[13];
	float posZ = transformation[8] * oriX + transformation[9] * oriY + transformation[10] * oriZ + transformation[14];
	const float deltaX = transformation[2] * volumeSize.z;
	const float deltaY = transformation[6] * volumeSize.z;
	const float deltaZ = transformation[10] * volumeSize.z;

	__syncthreads();

	for (int z = 0; z < resolution.z; z++) {
		bool flag = false;
		float tsdf = 1.0;

		posX += deltaX;
		posY += deltaY;
		posZ += deltaZ;

		float depthX = posX * DEPTH_FX / posZ + DEPTH_CX;
		float depthY = posY * DEPTH_FY / posZ + DEPTH_CY;
			
		if (posZ > 0 && 0 <= depthX && depthX <= DEPTH_W && 0 <= depthY && depthY <= DEPTH_H) {
			UINT16 depth = tex2D<UINT16>(depthTexture, depthX, depthY);

			if (depth != 0) {
				float sdf = depth * 0.001 - posZ;

				if (sdf >= -TRANC_DIST_M) {
					flag = true;
					sdf = sdf / TRANC_DIST_M;
					if (sdf < tsdf) {
						float colorX = posX * COLOR_FX / posZ + COLOR_CX;
						float colorY = posY * COLOR_FY / posZ + COLOR_CY;
						if ( 0 <= colorX && colorX <= COLOR_W && 0 <= colorY && colorY <= COLOR_H) {
							color = tex2D<uchar4>(colorTexture, colorX, colorY);
						}
						tsdf = sdf;
					}
				}
			}
		}

		int id = deviceVid(x, y, z, resolution);
		__syncthreads();

		if (flag) {
			volume[id] = tsdf;
		} else {
			volume[id] = -1;
		}
		__syncthreads();

		if (flag) {
			volume_color[id] = color;
		}
		__syncthreads();
	}
}

extern "C"
void cudaIntegrateDepth(int cameras, UINT16** depth, RGBQUAD** color, float** transformation) {
	HANDLE_ERROR(cudaMemcpy(transformation_device + 0 * 16, transformation[0], 16 * sizeof(float), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMemcpyToArray(depth_device, 0, 0, depth[0], sizeof(UINT16) * DEPTH_W * DEPTH_H, cudaMemcpyHostToDevice));
	depthTexture.filterMode = cudaFilterModePoint;
	depthTexture.addressMode[0] = cudaAddressModeWrap;
	depthTexture.addressMode[1] = cudaAddressModeWrap;
	HANDLE_ERROR(cudaBindTextureToArray(&depthTexture, depth_device, &depthDesc));

	HANDLE_ERROR(cudaMemcpyToArray(color_device, 0, 0, color[0], sizeof(uchar4) * COLOR_W * COLOR_H, cudaMemcpyHostToDevice));
	colorTexture.filterMode = cudaFilterModePoint;
	colorTexture.addressMode[0] = cudaAddressModeWrap;
	colorTexture.addressMode[1] = cudaAddressModeWrap;
	HANDLE_ERROR(cudaBindTextureToArray(&colorTexture, color_device, &colorDesc));

	kernelIntegrateDepth << <grid, block >> > (cameras, volume_device, volume_color_device, transformation_device, resolution, volumeSize, offset);
	HANDLE_ERROR(cudaGetLastError());

	HANDLE_ERROR(cudaUnbindTexture(&depthTexture));
	HANDLE_ERROR(cudaUnbindTexture(&colorTexture));
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
	if (x + 1 >= resolution.x) return 0;
	if (y + 1 >= resolution.y) return 0;
	if (z + 1 >= resolution.z) return 0;
	if (volume[deviceVid(x + 0, y + 0, z + 0, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 1, y + 0, z + 0, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 0, y + 1, z + 0, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 1, y + 1, z + 0, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 0, y + 0, z + 1, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 1, y + 0, z + 1, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 0, y + 1, z + 1, resolution)] == -1) return 0;
	if (volume[deviceVid(x + 1, y + 1, z + 1, resolution)] == -1) return 0;
	UINT16 index = 0;
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

__global__ void kernelMarchingCubesCount(float* volume, UINT8* cubeIndex, int* count, int3 resolution) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int cnt = 0;
	for (int z = 0; z + 1 < resolution.z; z++) {
		int id = deviceVid(x, y, z, resolution);
		int cubeId = cubeIndex[id] = deviceGetCubeIndex(volume, x, y, z, resolution);
		cnt += triNumber_device[cubeId];
		__syncthreads();
	}

	__syncthreads();
	count[devicePid(x, y, resolution)] = cnt;
}

__device__ __forceinline__ void deviceCalnEdgePoint(float* volume, uchar4* volume_color, int x1, int y1, int z1, int x2, int y2, int z2, float3& pos, uchar4& color, int3 resolution, float3 volumeSize, float3 offset) {
	int id1 = deviceVid(x1, y1, z1, resolution);
	int id2 = deviceVid(x2, y2, z2, resolution);
	float v1 = volume[id1];
	float v2 = volume[id2];
	if ((v1 < 0) ^ (v2 < 0)) {
		float k =  v1 / (v1 - v2);
		pos.x = ((1 - k) * x1 + k * x2 - 0.5) * volumeSize.x + offset.x;
		pos.y = ((1 - k) * y1 + k * y2 - 0.5) * volumeSize.y + offset.y;
		pos.z = ((1 - k) * z1 + k * z2 - 0.5) * volumeSize.z + offset.z;
		color.x = (UINT8)min((1 - k) * volume_color[id1].x + k * volume_color[id2].x, 255.0);
		color.y = (UINT8)min((1 - k) * volume_color[id1].y + k * volume_color[id2].y, 255.0);
		color.z = (UINT8)min((1 - k) * volume_color[id1].z + k * volume_color[id2].z, 255.0);
	}
}

__global__ void kernelMarchingCubes(float* volume, uchar4* volume_color, UINT8* cubeIndex, int* count, Vertex* vertex, int3 resolution, float3 volumeSize, float3 offset) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x + 1 >= resolution.x || y + 1 >= resolution.y) {
		return;
	}	

	float3 pos[12];
	uchar4 color[12];

	const int MAX_BUFFER = 512;
	int tot = 0;
	float3 posBuffer[MAX_BUFFER];
	uchar4 colorBuffer[MAX_BUFFER];

	for (int z = 0; z + 1 < resolution.z; z++) {
		int id = deviceVid(x, y, z, resolution);
		int cubeId = cubeIndex[id];

		deviceCalnEdgePoint(volume, volume_color, x + 0, y + 0, z + 0, x + 1, y + 0, z + 0, pos[0], color[0], resolution, volumeSize, offset);
		deviceCalnEdgePoint(volume, volume_color, x + 1, y + 0, z + 0, x + 1, y + 1, z + 0, pos[1], color[1], resolution, volumeSize, offset);
		deviceCalnEdgePoint(volume, volume_color, x + 1, y + 1, z + 0, x + 0, y + 1, z + 0, pos[2], color[2], resolution, volumeSize, offset);
		deviceCalnEdgePoint(volume, volume_color, x + 0, y + 1, z + 0, x + 0, y + 0, z + 0, pos[3], color[3], resolution, volumeSize, offset);
		
		deviceCalnEdgePoint(volume, volume_color, x + 0, y + 0, z + 1, x + 1, y + 0, z + 1, pos[4], color[4], resolution, volumeSize, offset);
		deviceCalnEdgePoint(volume, volume_color, x + 1, y + 0, z + 1, x + 1, y + 1, z + 1, pos[5], color[5], resolution, volumeSize, offset);
		deviceCalnEdgePoint(volume, volume_color, x + 1, y + 1, z + 1, x + 0, y + 1, z + 1, pos[6], color[6], resolution, volumeSize, offset);
		deviceCalnEdgePoint(volume, volume_color, x + 0, y + 1, z + 1, x + 0, y + 0, z + 1, pos[7], color[7], resolution, volumeSize, offset);

		deviceCalnEdgePoint(volume, volume_color, x + 0, y + 0, z + 0, x + 0, y + 0, z + 1, pos[8], color[8], resolution, volumeSize, offset);
		deviceCalnEdgePoint(volume, volume_color, x + 1, y + 0, z + 0, x + 1, y + 0, z + 1, pos[9], color[9], resolution, volumeSize, offset);
		deviceCalnEdgePoint(volume, volume_color, x + 1, y + 1, z + 0, x + 1, y + 1, z + 1, pos[10], color[10], resolution, volumeSize, offset);
		deviceCalnEdgePoint(volume, volume_color, x + 0, y + 1, z + 0, x + 0, y + 1, z + 1, pos[11], color[11], resolution, volumeSize, offset);
		
		if (triTable_device[cubeId][0] != -1) {
			for (int i = 0; i < 5; i++) {
				if (triTable_device[cubeId][i * 3] != -1) {
					for (int j = 0; j < 3; j++) {
						int edgeId = triTable_device[cubeId][i * 3 + j];
						posBuffer[tot] = pos[edgeId];
						colorBuffer[tot] = color[edgeId];
						tot++;
					}
				} else {
					break;
				}
			}
		}

		__syncthreads();
	}

	int tid = count[devicePid(x, y, resolution)] * 3;
	for (int i = 0; i < tot; i++) {
		vertex[tid + i].pos = posBuffer[i];
		vertex[tid + i].color = colorBuffer[i];
	}
}

__global__ void cudaCountAccumulation(int *count_device, int *sum_device, int *temp_device) {//一个block用1024个线程，处理2048个数。一共需要处理resx*resy = 2^18个数，分成128个block。fixed！！
	int block_offset = blockIdx.x * 2048;//确定处理第几个2048。
	int thid = threadIdx.x;

	__shared__ int shared_count_device[2048];
	if (block_offset == 0 && thid == 0)
		shared_count_device[0] = 0;
	else
		shared_count_device[2 * thid] = count_device[block_offset + 2 * thid - 1];
	shared_count_device[2 * thid + 1] = count_device[block_offset + 2 * thid];
	__syncthreads();//shared赋值需要同步。

					//UpSweep
	int offset = 1;
	int n = 2048;
	for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			shared_count_device[bi] += shared_count_device[ai];
		}
		offset *= 2;
	}

	//DownSweep,注意这里因为要和其它block再求和，所以使用的是inclusive scan，不是教程中的exclusive scan。
	for (int i = n / 2; i > 1; i /= 2) {
		__syncthreads();
		int start = (i - 1) + (i >> 1);
		int doffset = (i >> 1);
		if ((2 * thid - start) % i == 0 && 2 * thid - start >= 0) {
			shared_count_device[2 * thid] += shared_count_device[2 * thid - doffset];
		}
		if ((2 * thid + 1 - start) % i == 0 && 2 * thid + 1 - start >= 0) {
			shared_count_device[2 * thid + 1] += shared_count_device[2 * thid + 1 - doffset];
		}
	}
	temp_device[block_offset + 2 * thid] = shared_count_device[2 * thid];
	temp_device[block_offset + 2 * thid + 1] = shared_count_device[2 * thid + 1];
	if (thid == 0) {
		sum_device[blockIdx.x] = shared_count_device[n - 1];
	}
}

__global__ void cudaCountAccumulation2(int *count_device, int *sum_device, int *temp_device) {//一个block用1024个线程，处理2048个数。一共需要处理resx*resy = 2^18个数，分成128个block。fixed！！
	int block_offset = blockIdx.x * 2048;//确定处理第几个2048。
	int thid = threadIdx.x;
	int n = 2048;
	__shared__ int shared_count_device[2048];
	__shared__ int presum;
	shared_count_device[2 * thid] = temp_device[block_offset + 2 * thid];
	shared_count_device[2 * thid + 1] = temp_device[block_offset + 2 * thid + 1];
	if (blockIdx.x == 0 && thid == 0) {
		count_device[0] = count_device[114688];
	}
	if (thid == 0) {
		if (blockIdx.x != 0) {
			presum = sum_device[blockIdx.x - 1];
		}
		else {
			presum = 0;
		}
	}
	__syncthreads();//shared赋值需要同步。

	shared_count_device[2 * thid] += presum;
	shared_count_device[2 * thid + 1] += presum;

	count_device[block_offset + 2 * thid] = shared_count_device[2 * thid];
	count_device[block_offset + 2 * thid + 1] = shared_count_device[2 * thid + 1];
}

int cpu_cudaCountAccumulation() {
	int temp_grid = 128, temp_block = 1024;
	const int DATASIZE = 262144;
	int *temp_device;
	int *sum_device, *sum_host;
	HANDLE_ERROR(cudaMalloc(&temp_device, DATASIZE * sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&sum_device, temp_grid * sizeof(int)));
	sum_host = new int[temp_grid];
	for (int i = 0; i < temp_grid; ++i) {
		sum_host[i] = 0;
	}
	//stage1
	HANDLE_ERROR(cudaMemcpy(sum_device, sum_host, temp_grid * sizeof(int), cudaMemcpyHostToDevice));
	cudaCountAccumulation << <temp_grid, temp_block >> > (count_device, sum_device, temp_device);
	HANDLE_ERROR(cudaMemcpy(sum_host, sum_device, temp_grid * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 1; i<temp_grid; ++i) {
		sum_host[i] += sum_host[(i - 1)];
	}
	//stage2
	HANDLE_ERROR(cudaMemcpy(sum_device, sum_host, temp_grid * sizeof(int), cudaMemcpyHostToDevice));
	cudaCountAccumulation2 << <temp_grid, temp_block >> > (count_device, sum_device, temp_device);

	int tris_size = sum_host[temp_grid - 1];
	return tris_size;
}

extern "C"
void cudaCalculateMesh(Vertex* vertex, int& tri_size) {
	kernelMarchingCubesCount << <grid, block >> > (volume_device, cubeIndex_device, count_device, resolution);

	cudaThreadSynchronize();
	Timer timer;

	HANDLE_ERROR(cudaGetLastError());
	tri_size = cpu_cudaCountAccumulation();
	HANDLE_ERROR(cudaGetLastError());

	cudaThreadSynchronize();
	timer.outputTime();

	Vertex* vertex_device;
	HANDLE_ERROR(cudaMalloc(&vertex_device, tri_size * 3 * sizeof(Vertex)));
	kernelMarchingCubes << <grid, block >> > (volume_device, volume_color_device, cubeIndex_device, count_device, vertex_device, resolution, volumeSize, offset);

	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaMemcpy(vertex, vertex_device, tri_size * 3 * sizeof(Vertex), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(vertex_device));
}
