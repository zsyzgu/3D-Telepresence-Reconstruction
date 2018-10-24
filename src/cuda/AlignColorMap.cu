#include "AlignColorMap.h"
#include "CudaHandleError.h"
#include "Parameters.h"

namespace BackgroundNamespace {
	__constant__ float COLOR_THRESHOLD = 50.0f;
	__constant__ float DEPTH_THRESHOLD = 0.05f;
};
using namespace BackgroundNamespace;

__global__ void kernelAlignProcess(uchar4* alignedColor, float* depth, uchar4* color, Intrinsics depthIntrinsics, Intrinsics colorIntrinsics, Transformation depth2color) {
	const int MAX_SHIFT = DEPTH_W >> 4;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int2 colorPixel_shared[COLOR_W];

	for (int i = threadIdx.x; i < COLOR_W; i += blockDim.x) {
		float2 pixelFloat = make_float2((float)i * DEPTH_W / COLOR_W, (float)y * DEPTH_H / COLOR_H);
		int2 pixel = make_int2((int)pixelFloat.x, (int)pixelFloat.y);

		if (0 <= pixel.x && pixel.x < DEPTH_W && 0 <= pixel.y && pixel.y < DEPTH_H) {
			float3 pos = depthIntrinsics.deproject(pixelFloat, depth[pixel.y * DEPTH_W + pixel.x]);
			pos = depth2color.translate(pos);
			int2 colorPixel = colorIntrinsics.translate(pos);
			colorPixel_shared[i] = colorPixel;
		} else {
			colorPixel_shared[i] = make_int2(-1, -1);
		}
	}
	__syncthreads();

	if (x < COLOR_W && y < COLOR_H) {
		uchar4 result = uchar4();
		int2 colorPixel = colorPixel_shared[x];

		if (0 <= colorPixel.x && colorPixel.x < COLOR_W && 0 <= colorPixel.y && colorPixel.y < COLOR_H) {
			result = color[colorPixel.y * COLOR_W + colorPixel.x];
		}

		for (int shift = 1; shift <= MAX_SHIFT; shift++) {
			if (x - shift >= 0 && colorPixel_shared[x - shift].x > colorPixel.x) {
				result = uchar4();
				break;
			}
		}

		__syncthreads();
		alignedColor[y * COLOR_W + x] = result;
	}
}

__global__ void kernelRemoveBackground(uchar4* color, float* depth, uchar4* colorBackground, float* depthBackground) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < DEPTH_W && y < DEPTH_H) {
		int id = y * DEPTH_W + x;
		if (depth[id] != 0 && depthBackground[id] != 0 && fabs(depth[id] - depthBackground[id]) < DEPTH_THRESHOLD) {
			int cx = x * COLOR_W / DEPTH_W;
			int cy = y * COLOR_H / DEPTH_H;
			if (cx < COLOR_W && cy < COLOR_H) {
				int cid = cy * COLOR_W + cx;
				uchar4 c0 = color[cid];
				uchar4 c1 = colorBackground[cid];
				float colorDiff = (float)(abs(c0.x - c1.x) + abs(c0.y - c1.y) + abs(c0.z - c1.z)) / 3;
				if (colorDiff < COLOR_THRESHOLD) {
					depth[id] = 0;
				}
			}
		}
	}
}

extern "C"
void cudaAlignInit(RGBQUAD *& alignedColor_device, float *& depthBackground_device, RGBQUAD *& colorBackground_device) {
	HANDLE_ERROR(cudaMalloc(&alignedColor_device, MAX_CAMERAS * COLOR_H * COLOR_W * sizeof(RGBQUAD)));
	HANDLE_ERROR(cudaMalloc(&depthBackground_device, MAX_CAMERAS * DEPTH_H * DEPTH_W * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&colorBackground_device, MAX_CAMERAS * COLOR_H * COLOR_W * sizeof(RGBQUAD)));
}

extern "C"
void cudaAlignClean(RGBQUAD *& alignedColor_device, float *& depthBackground_device, RGBQUAD *& colorBackground_device) {
	HANDLE_ERROR(cudaFree(alignedColor_device));
	HANDLE_ERROR(cudaFree(depthBackground_device));
	HANDLE_ERROR(cudaFree(colorBackground_device));
}

extern "C"
void cudaAlignProcess(int cameras, RGBQUAD* alignedColor_device, float* depth_device, RGBQUAD* color_device, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics, Transformation* depth2color) {
	dim3 threadsPerBlock = dim3(512, 1);
	dim3 blocksPerGrid = dim3((COLOR_W + threadsPerBlock.x - 1) / threadsPerBlock.x, (COLOR_H + threadsPerBlock.y - 1) / threadsPerBlock.y);
	
	for (int i = 0; i < cameras; i++) {
		kernelAlignProcess << <blocksPerGrid, threadsPerBlock >> >((uchar4*)alignedColor_device + i * COLOR_H * COLOR_W, depth_device + i * DEPTH_H * DEPTH_W, (uchar4*)color_device + i * COLOR_H * COLOR_W, depthIntrinsics[i], colorIntrinsics[i], depth2color[i]);
		cudaGetLastError();
	}

	cudaThreadSynchronize();
}

extern "C"
void cudaRemoveBackground(int cameras, RGBQUAD* alignedColor_device, float* depth_device, RGBQUAD* colorBackground_device, float* depthBackground_device) {
	dim3 threadsPerBlock = dim3(256, 1);
	dim3 blocksPerGrid = dim3((DEPTH_W + threadsPerBlock.x - 1) / threadsPerBlock.x, (DEPTH_H + threadsPerBlock.y - 1) / threadsPerBlock.y);

	for (int i = 0; i < cameras; i++) {
		kernelRemoveBackground << <blocksPerGrid, threadsPerBlock >> > ((uchar4*)alignedColor_device + i * COLOR_H * COLOR_W, depth_device + i * DEPTH_H * DEPTH_W, (uchar4*)colorBackground_device + i * COLOR_H * COLOR_W, depthBackground_device + i * DEPTH_H * DEPTH_W);
		cudaGetLastError();
	}
}
