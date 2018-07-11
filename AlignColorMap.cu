#include "AlignColorMap.h"
#include "CudaHandleError.h"
#include "Parameters.h"

__global__ void kernelAlignProcess(uchar4* alignedColor, float* depth, uchar4* color, Intrinsics depthIntrinsics, Intrinsics colorIntrinsics, Transformation depth2color) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < COLOR_W && y < COLOR_H) {
		float2 dpFloat = make_float2((float)x * DEPTH_W / COLOR_W, (float)y * DEPTH_H / COLOR_H);
		int2 dp = make_int2((int)dpFloat.x, (int)dpFloat.y);
		uchar4 result = make_uchar4(0, 0, 0, 0);
		if (0 <= dp.x && dp.x < DEPTH_W && 0 <= dp.y && dp.y < DEPTH_H) {
			float z = depth[dp.y * DEPTH_W + dp.x];
			float3 pos = depthIntrinsics.deproject(dpFloat, z);
			pos = depth2color.translate(pos);
			int2 cp = colorIntrinsics.translate(pos);
			if (0 <= cp.x && cp.x < COLOR_W && 0 <= cp.y && cp.y < COLOR_H) {
				result = color[cp.y * COLOR_W + cp.x];
			}
		}
		alignedColor[y * COLOR_W + x] = result;
	}
}

extern "C"
void cudaAlignInit(RGBQUAD*& alignedColor_device) {
	HANDLE_ERROR(cudaMalloc(&alignedColor_device, COLOR_H * COLOR_W * sizeof(RGBQUAD)));
}

extern "C"
void cudaAlignClean(RGBQUAD*& alignedColor_device) {
	HANDLE_ERROR(cudaFree(alignedColor_device));
}

extern "C"
void cudaAlignProcess(int cameras, RGBQUAD* alignedColor_device, float* depth_device, RGBQUAD* color_device, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics, Transformation* depth2color) {
	dim3 threadsPerBlock = dim3(256, 1);
	dim3 blocksPerGrid = dim3((COLOR_W + threadsPerBlock.x - 1) / threadsPerBlock.x, (COLOR_H + threadsPerBlock.y - 1) / threadsPerBlock.y);
	
	for (int i = 0; i < cameras; i++) {
		kernelAlignProcess<<<blocksPerGrid, threadsPerBlock>>>((uchar4*)alignedColor_device + i * COLOR_H * COLOR_W, depth_device + i * DEPTH_H * DEPTH_W, (uchar4*)color_device + i * COLOR_H * COLOR_W, depthIntrinsics[i], colorIntrinsics[i], depth2color[i]);
		cudaGetLastError();
	}
	cudaThreadSynchronize();
}