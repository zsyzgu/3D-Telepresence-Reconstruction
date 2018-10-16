#include "AlignColorMap.h"

extern "C" void cudaAlignInit(RGBQUAD*& alignedColor_device, float*& depthBackground_device, RGBQUAD*& colorBackground_device);
extern "C" void cudaAlignClean(RGBQUAD*& alignedColor_device, float*& depthBackground_device, RGBQUAD*& colorBackground_device);
extern "C" void cudaAlignProcess(int cameras, bool* check, RGBQUAD* alignedColor_device, float* depth_device, RGBQUAD* color_device, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics, Transformation* depth2color);
extern "C" void cudaRemoveBackground(int cameras, bool* check, RGBQUAD* alignedColor_device, float* depth_device, RGBQUAD* colorBackground_device, float* depthBackground_device);

AlignColorMap::AlignColorMap()
{
	isRemoveBackground = false;
	cudaAlignInit(alignedColor_devive, depthBackground_device, colorBackground_device);
}

AlignColorMap::~AlignColorMap()
{
	cudaAlignClean(alignedColor_devive, depthBackground_device, colorBackground_device);
}

void AlignColorMap::alignColor2Depth(int cameras, bool * check, float * depth_device, RGBQUAD * color_device, Intrinsics * depthIntrinsics, Intrinsics * colorIntrinsics, Transformation * depth2color)
{
	cudaAlignProcess(cameras, check, alignedColor_devive, depth_device, color_device, depthIntrinsics, colorIntrinsics, depth2color);
	if (isRemoveBackground) {
		cudaRemoveBackground(cameras, check, alignedColor_devive, depth_device, colorBackground_device, depthBackground_device);
	}
}

void AlignColorMap::enableBackground()
{
	isRemoveBackground = true;
}

void AlignColorMap::enableBackground(float* depth_device) {
	isRemoveBackground = true;
	HANDLE_ERROR(cudaMemcpy(depthBackground_device, depth_device, MAX_CAMERAS * DEPTH_W * DEPTH_H * sizeof(float), cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaMemcpy(colorBackground_device, alignedColor_devive, MAX_CAMERAS * COLOR_W * COLOR_H * sizeof(RGBQUAD), cudaMemcpyDeviceToDevice));
}

void AlignColorMap::disableBackground()
{
	isRemoveBackground = false;
}

void AlignColorMap::copyBackground_host2device(float* depthBackground, RGBQUAD* colorBackground) {
	HANDLE_ERROR(cudaMemcpy(depthBackground_device, depthBackground, MAX_CAMERAS * DEPTH_W * DEPTH_H * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(colorBackground_device, colorBackground, MAX_CAMERAS * COLOR_W * COLOR_H * sizeof(RGBQUAD), cudaMemcpyHostToDevice));
}

void AlignColorMap::copyBackground_device2host(float* depthBackground, RGBQUAD* colorBackground) {
	HANDLE_ERROR(cudaMemcpy(depthBackground, depthBackground_device, MAX_CAMERAS * DEPTH_W * DEPTH_H * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(colorBackground, colorBackground_device, MAX_CAMERAS * COLOR_W * COLOR_H * sizeof(RGBQUAD), cudaMemcpyDeviceToHost));
}

