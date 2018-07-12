#include "AlignColorMap.h"
#include "CudaHandleError.h"
#include "Parameters.h"

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

extern "C"
void cudaAlignInit(RGBQUAD*& alignedColor_device) {
	HANDLE_ERROR(cudaMalloc(&alignedColor_device, MAX_CAMERAS * COLOR_H * COLOR_W * sizeof(RGBQUAD)));
}

extern "C"
void cudaAlignClean(RGBQUAD*& alignedColor_device) {
	HANDLE_ERROR(cudaFree(alignedColor_device));
}

extern "C"
void cudaAlignProcess(int cameras, RGBQUAD* alignedColor_device, float* depth_device, RGBQUAD* color_device, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics, Transformation* depth2color) {
	dim3 threadsPerBlock = dim3(512, 1);
	dim3 blocksPerGrid = dim3((COLOR_W + threadsPerBlock.x - 1) / threadsPerBlock.x, (COLOR_H + threadsPerBlock.y - 1) / threadsPerBlock.y);
	
	for (int i = 0; i < cameras; i++) {
		kernelAlignProcess<<<blocksPerGrid, threadsPerBlock>>>((uchar4*)alignedColor_device + i * COLOR_H * COLOR_W, depth_device + i * DEPTH_H * DEPTH_W, (uchar4*)color_device + i * COLOR_H * COLOR_W, depthIntrinsics[i], colorIntrinsics[i], depth2color[i]);
		cudaGetLastError();
	}
	cudaThreadSynchronize();
}