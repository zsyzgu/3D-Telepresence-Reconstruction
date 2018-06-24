#include "CudaHandleError.h"
#include "Parameters.h"
#include <Windows.h>
#include <iostream>
#include "Timer.h"
#include "Vertex.h"
#include "Parameters.h"
#include "TsdfVolume.cuh"

namespace tsdf {
	int3 resolution;
	float3 size;
	float3 center;
	float3 volumeSize;
	float3 offset;

	float* volume_device;
	UINT8* volumeBin_device;
	cudaArray** depth_device;
	cudaArray** color_device;
	Transformation* depthTrans_device;
	Transformation* colorTrans_device;
	Intrinsics* depthIntrinsics_device;
	Intrinsics* colorIntrinsics_device;
	int* count_device;
	int* count_host;

	dim3 grid;
	dim3 block;
}
using namespace tsdf;

#if VOLUME == 256
CUDA_CALLABLE_MEMBER __forceinline__ int devicePid(int x, int y) {
	return (x & 15) | ((y & 15) << 4) | ((x >> 4) << 8) | ((y >> 4) << 12);
}

CUDA_CALLABLE_MEMBER __forceinline__ int deviceVid(int x, int y, int z) {
	return (x & 15) | ((y & 15) << 4) | ((z & 15) << 8) | ((x >> 4) << 12) | ((y >> 4) << 16) | ((z >> 4) << 20);
}
#elif VOLUME == 512
CUDA_CALLABLE_MEMBER __forceinline__ int devicePid(int x, int y) {
	return (x & 15) | ((y & 15) << 4) | ((x >> 4) << 8) | ((y >> 4) << 13);
}

CUDA_CALLABLE_MEMBER __forceinline__ int deviceVid(int x, int y, int z) {
	return (x & 15) | ((y & 15) << 4) | ((z & 15) << 8) | ((x >> 4) << 12) | ((y >> 4) << 17) | ((z >> 4) << 22);
}
#endif


extern "C"
void cudaInitVolume(float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ) {
	size = make_float3(sizeX, sizeY, sizeZ);
	center = make_float3(centerX, centerY, centerZ);
	volumeSize = size * (1.0 / VOLUME);
	offset = center - size * 0.5;
	HANDLE_ERROR(cudaMalloc(&volume_device, VOLUME * VOLUME * VOLUME * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&volumeBin_device, VOLUME * VOLUME * VOLUME * sizeof(UINT8)));

	cudaChannelFormatDesc depthDesc = cudaCreateChannelDesc<UINT16>();
	cudaChannelFormatDesc colorDesc = cudaCreateChannelDesc<uchar4>();
	depth_device = new cudaArray*[MAX_CAMERAS];
	color_device = new cudaArray*[MAX_CAMERAS];
	cudaTextureDesc texDesc;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	cudaResourceDesc resDesc;
	resDesc.resType = cudaResourceTypeArray;
	cudaTextureObject_t texObj;
	for (int i = 0; i < MAX_CAMERAS; i++) {
		HANDLE_ERROR(cudaMallocArray(&depth_device[i], &depthDesc, DEPTH_W, DEPTH_H));
		resDesc.res.array.array = depth_device[i];
		cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	}
	for (int i = 0; i < MAX_CAMERAS; i++) {
		HANDLE_ERROR(cudaMallocArray(&color_device[i], &colorDesc, COLOR_W, COLOR_H));
		resDesc.res.array.array = color_device[i];
		cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	}

	HANDLE_ERROR(cudaMalloc(&depthTrans_device, MAX_CAMERAS * sizeof(Transformation)));
	HANDLE_ERROR(cudaMalloc(&colorTrans_device, MAX_CAMERAS * sizeof(Transformation)));
	HANDLE_ERROR(cudaMalloc(&depthIntrinsics_device, MAX_CAMERAS * sizeof(Intrinsics)));
	HANDLE_ERROR(cudaMalloc(&colorIntrinsics_device, MAX_CAMERAS * sizeof(Intrinsics)));

	HANDLE_ERROR(cudaMalloc(&count_device, VOLUME * VOLUME * sizeof(int)));
	count_host = new int[VOLUME * VOLUME];
}

extern "C"
void cudaReleaseVolume() {
	HANDLE_ERROR(cudaFree(volume_device));
	HANDLE_ERROR(cudaFree(volumeBin_device));
	for (int i = 0; i < MAX_CAMERAS; i++) {
		HANDLE_ERROR(cudaFreeArray(depth_device[i]));
		HANDLE_ERROR(cudaFreeArray(color_device[i]));
	}
	if (depth_device != NULL) {
		delete[] depth_device;
	}
	if (color_device != NULL) {
		delete[] color_device;
	}
	for (int i = 0; i < MAX_CAMERAS; i++) {
		HANDLE_ERROR(cudaDestroyTextureObject(i + 1));
		HANDLE_ERROR(cudaDestroyTextureObject(i + MAX_CAMERAS + 1));
	}

	HANDLE_ERROR(cudaFree(depthTrans_device));
	HANDLE_ERROR(cudaFree(colorTrans_device));
	HANDLE_ERROR(cudaFree(depthIntrinsics_device));
	HANDLE_ERROR(cudaFree(colorIntrinsics_device));

	HANDLE_ERROR(cudaFree(count_device));
	delete[] count_host;
}

__global__ void kernelIntegrateDepth(int cameras, float* volume, UINT8* volumeBin, Transformation* transformation, Intrinsics* intrinsics, float3 volumeSize, float3 offset) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= VOLUME || y >= VOLUME) {
		return;
	}

	const float TRANC_DIST_M = 2.1 * max(volumeSize.x, max(volumeSize.y, volumeSize.z));

	float3 ori = make_float3(x, y, -1) * volumeSize + offset;
	float3 pos[MAX_CAMERAS];
	float3 deltaZ[MAX_CAMERAS];

	for (int i = 0; i < cameras; i++) {
		pos[i] = transformation[i].translate(ori);
		deltaZ[i] = transformation[i].deltaZ() * volumeSize;
	}

	for (int z = 0; z < VOLUME; z++) {
		int cnt = 0;
		UINT8 bin = 0;
		float tsdf = 0;

		for (int i = 0; i < cameras; i++) {
			pos[i] = pos[i] + deltaZ[i];
			float2 pixel = intrinsics[i].translate(pos[i]);

			if (pos[i].z > 0 && 0 <= pixel.x && pixel.x <= DEPTH_W && 0 <= pixel.y && pixel.y <= DEPTH_H) {
				UINT16 depth = tex2D<UINT16>(i + 1, pixel.x, pixel.y);

				if (depth != 0) {
					float sdf = depth * 0.001 - pos[i].z;

					if (sdf >= -TRANC_DIST_M) {
						cnt++;
						bin |= (1 << i);
						tsdf += sdf / TRANC_DIST_M;
					}
				}
			}
		}

		int id = deviceVid(x, y, z);
		if (cnt != 0) {
			volume[id] = tsdf / cnt;
		} else {
			volume[id] = -1;
		}
		volumeBin[id] = bin;
	}
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

__device__ __forceinline__ UINT16 deviceGetCubeIndex(float* volume, int x, int y, int z) {
	if (x + 1 >= VOLUME) return 0;
	if (y + 1 >= VOLUME) return 0;
	if (z + 1 >= VOLUME) return 0;
	if (volume[deviceVid(x + 0, y + 0, z + 0)] == -1) return 0;
	if (volume[deviceVid(x + 1, y + 0, z + 0)] == -1) return 0;
	if (volume[deviceVid(x + 0, y + 1, z + 0)] == -1) return 0;
	if (volume[deviceVid(x + 1, y + 1, z + 0)] == -1) return 0;
	if (volume[deviceVid(x + 0, y + 0, z + 1)] == -1) return 0;
	if (volume[deviceVid(x + 1, y + 0, z + 1)] == -1) return 0;
	if (volume[deviceVid(x + 0, y + 1, z + 1)] == -1) return 0;
	if (volume[deviceVid(x + 1, y + 1, z + 1)] == -1) return 0;
	UINT16 index = 0;
	if (volume[deviceVid(x + 0, y + 0, z + 0)] < 0) index |= 1;
	if (volume[deviceVid(x + 1, y + 0, z + 0)] < 0) index |= 2;
	if (volume[deviceVid(x + 0, y + 1, z + 0)] < 0) index |= 8;
	if (volume[deviceVid(x + 1, y + 1, z + 0)] < 0) index |= 4;
	if (volume[deviceVid(x + 0, y + 0, z + 1)] < 0) index |= 16;
	if (volume[deviceVid(x + 1, y + 0, z + 1)] < 0) index |= 32;
	if (volume[deviceVid(x + 0, y + 1, z + 1)] < 0) index |= 128;
	if (volume[deviceVid(x + 1, y + 1, z + 1)] < 0) index |= 64;
	return index;
}

__global__ void kernelMarchingCubesCount(float* volume, int* count) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int cnt = 0;
	for (int z = 0; z + 1 < VOLUME; z++) {
		cnt += triNumber_device[deviceGetCubeIndex(volume, x, y, z)];
	}
	count[devicePid(x, y)] = cnt;
}

__device__ __forceinline__ float3 deviceCalnEdgePoint(float* volume, int x, int y, int z, int dx, int dy, int dz) {
	float v1 = volume[deviceVid(x, y, z)];
	float v2 = volume[deviceVid(x + dx, y + dy, z + dz)];
	if ((v1 < 0) ^ (v2 < 0)) {
		float k =  v1 / (v1 - v2);
		return make_float3(x + k * dx, y + k * dy, z + k * dz);
	}
	return float3();
}

__device__ __forceinline__ uchar4 calnColor(int cameras, UINT8 bin, float3 ori, Transformation* transformation, Intrinsics* intrinsics) {
	short4 colorSum = short4();
	int cnt = 0;
	for (int i = 0; i < cameras; i++) {
		if ((bin >> i) & 1) {
			float3 pos = transformation[i].translate(ori);
			float2 pixel = intrinsics[i].translate(pos);
			if (0 <= pixel.x && pixel.x <= COLOR_W && 0 <= pixel.y && pixel.y <= COLOR_H) {
				cnt++;
				uchar4 tmp = tex2D<uchar4>(i + MAX_CAMERAS + 1, pixel.x, pixel.y);
				colorSum.x += tmp.x;
				colorSum.y += tmp.y;
				colorSum.z += tmp.z;
			}
		}
	}
	if (cnt == 0) {
		return uchar4();
	}
	return make_uchar4(colorSum.x / cnt, colorSum.y / cnt, colorSum.z / cnt, 0);
}

__global__ void kernelMarchingCubes(int cameras, float* volume, UINT8* volumeBin, int* count, Vertex* vertex, Transformation* transformation, Intrinsics* intrinsics, float3 volumeSize, float3 offset) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x + 1 >= VOLUME || y + 1 >= VOLUME) {
		return;
	}	

	float3 pos[12];
	float3 posBuffer[6];
	Vertex* vtx = vertex + count[devicePid(x, y)] * 3;

	for (int z = 0; z + 1 < VOLUME; z++) {
		int cubeId = deviceGetCubeIndex(volume, x, y, z);

		if (triTable_device[cubeId][0] != -1) {
			int id = deviceVid(x, y, z);
			pos[0] = deviceCalnEdgePoint(volume, x + 0, y + 0, z + 0, 1, 0, 0);
			pos[1] = deviceCalnEdgePoint(volume, x + 1, y + 0, z + 0, 0, 1, 0);
			pos[2] = deviceCalnEdgePoint(volume, x + 0, y + 1, z + 0, 1, 0, 0);
			pos[3] = deviceCalnEdgePoint(volume, x + 0, y + 0, z + 0, 0, 1, 0);

			pos[4] = deviceCalnEdgePoint(volume, x + 0, y + 0, z + 1, 1, 0, 0);
			pos[5] = deviceCalnEdgePoint(volume, x + 1, y + 0, z + 1, 0, 1, 0);
			pos[6] = deviceCalnEdgePoint(volume, x + 0, y + 1, z + 1, 1, 0, 0);
			pos[7] = deviceCalnEdgePoint(volume, x + 0, y + 0, z + 1, 0, 1, 0);

			pos[8] = deviceCalnEdgePoint(volume, x + 0, y + 0, z + 0, 0, 0, 1);
			pos[9] = deviceCalnEdgePoint(volume, x + 1, y + 0, z + 0, 0, 0, 1);
			pos[10] = deviceCalnEdgePoint(volume, x + 1, y + 1, z + 0, 0, 0, 1);
			pos[11] = deviceCalnEdgePoint(volume, x + 0, y + 1, z + 0, 0, 0, 1);

			for (int i = 0; i < 5 && triTable_device[cubeId][i * 3] != -1; i++) {
				for (int j = 0; j < 3; j++) {
					int edgeId = triTable_device[cubeId][i * 3 + j];
					posBuffer[j] = pos[edgeId] * volumeSize + offset;
				}
				posBuffer[3] = (posBuffer[0] + posBuffer[1]) * 0.5;
				posBuffer[4] = (posBuffer[1] + posBuffer[2]) * 0.5;
				posBuffer[5] = (posBuffer[2] + posBuffer[0]) * 0.5;
				for (int j = 0; j < 3; j++) {
					vtx->pos = posBuffer[j];
					vtx->color = calnColor(cameras, volumeBin[id], posBuffer[j], transformation, intrinsics);
					vtx->color2 = calnColor(cameras, volumeBin[id], posBuffer[j + 3], transformation, intrinsics);
					vtx++;
				}
			}
		}
	}
}

__global__ void cudaCountAccumulation(int *count_device, int *sum_device, int *temp_device) {//一个block用1024个线程，处理2048个数。一共需要处理resx*resy = 2^18个数，分成128个block。
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

	//DownSweep,注意这里因为要和其它block再求和，所以使用的是inclusive scan。
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

__global__ void cudaCountAccumulation2(int *count_device, int *sum_device, int *temp_device) {//一个block用1024个线程，处理2048个数。一共需要处理resx*resy = 2^18个数，分成128个block。
	int block_offset = blockIdx.x * 2048;//确定处理第几个2048。
	int thid = threadIdx.x;
	int n = 2048;
	__shared__ int shared_count_device[2048];
	__shared__ int presum;
	shared_count_device[2 * thid] = temp_device[block_offset + 2 * thid];
	shared_count_device[2 * thid + 1] = temp_device[block_offset + 2 * thid + 1];
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
	const int DATASIZE = VOLUME * VOLUME;
	int threads = 1024;
	int blocks = DATASIZE / threads / 2;
	static int sum_host[128];
	static int* sum_device = NULL;
	static int* temp_device = NULL;
	if (sum_device == NULL) {
		HANDLE_ERROR(cudaMalloc(&sum_device, blocks * sizeof(int)));
		HANDLE_ERROR(cudaMalloc(&temp_device, DATASIZE * sizeof(int)));
	}
	//stage1
	cudaCountAccumulation << <blocks, threads >> > (count_device, sum_device, temp_device);
	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaMemcpy(sum_host, sum_device, blocks * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 1; i < blocks; ++i) {
		sum_host[i] += sum_host[i - 1];
	}
	//stage2
	HANDLE_ERROR(cudaMemcpy(sum_device, sum_host, blocks * sizeof(int), cudaMemcpyHostToDevice));
	cudaCountAccumulation2 << <blocks, threads >> > (count_device, sum_device, temp_device);
	HANDLE_ERROR(cudaGetLastError());

	int tris_size = sum_host[blocks - 1];
	return tris_size;
}

extern "C"
void cudaIntegrate(int cameras, int& triSize, Vertex* vertex, UINT16** depth, RGBQUAD** color, Transformation* depthTrans, Transformation* colorTrans, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics) {
	//cudaThreadSynchronize();
	//Timer timer;
	dim3 blocks = dim3(VOLUME / BLOCK_SIZE, VOLUME / BLOCK_SIZE);
	dim3 threads = dim3(BLOCK_SIZE, BLOCK_SIZE);

	HANDLE_ERROR(cudaMemcpy(depthTrans_device, depthTrans, MAX_CAMERAS * sizeof(Transformation), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(colorTrans_device, colorTrans, MAX_CAMERAS * sizeof(Transformation), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(depthIntrinsics_device, depthIntrinsics, MAX_CAMERAS * sizeof(Intrinsics), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(colorIntrinsics_device, colorIntrinsics, MAX_CAMERAS * sizeof(Intrinsics), cudaMemcpyHostToDevice));

	for (int i = 0; i < cameras; i++) {
		HANDLE_ERROR(cudaMemcpyToArray(depth_device[i], 0, 0, depth[i], sizeof(UINT16) * DEPTH_W * DEPTH_H, cudaMemcpyHostToDevice));
	}
	for (int i = 0; i < cameras; i++) {
		HANDLE_ERROR(cudaMemcpyToArrayAsync(color_device[i], 0, 0, color[i], sizeof(uchar4) * COLOR_W * COLOR_H, cudaMemcpyHostToDevice));
	}

	kernelIntegrateDepth << <blocks, threads >> > (cameras, volume_device, volumeBin_device, depthTrans_device, depthIntrinsics_device, volumeSize, offset);
	HANDLE_ERROR(cudaGetLastError());

	kernelMarchingCubesCount << <blocks, threads >> > (volume_device, count_device);
	HANDLE_ERROR(cudaGetLastError());
	triSize = cpu_cudaCountAccumulation();

	Vertex* vertex_device;
	HANDLE_ERROR(cudaMalloc(&vertex_device, triSize * 3 * sizeof(Vertex)));
	kernelMarchingCubes << <blocks, threads >> > (cameras, volume_device, volumeBin_device, count_device, vertex_device, colorTrans_device, colorIntrinsics_device, volumeSize, offset);
	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaMemcpy(vertex, vertex_device, triSize * 3 * sizeof(Vertex), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(vertex_device));

	//cudaThreadSynchronize();
	//timer.outputTime();
}
