#include "AlignColorMap.h"

extern "C" void cudaAlignInit(RGBQUAD*& alignedColor_device);
extern "C" void cudaAlignClean(RGBQUAD*& alignedColor_device);
extern "C" void cudaAlignProcess(int cameras, RGBQUAD* alignedColor_device, float* depth_device, RGBQUAD* color_device, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics, Transformation* depth2color);

AlignColorMap::AlignColorMap()
{
	cudaAlignInit(alignedColor_devive);
}

AlignColorMap::~AlignColorMap()
{
	cudaAlignClean(alignedColor_devive);
}

RGBQUAD* AlignColorMap::getAlignedColor_device(int cameras, float* depth_device, RGBQUAD* color_device, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics, Transformation* depth2color)
{
	cudaAlignProcess(cameras, alignedColor_devive, depth_device, color_device, depthIntrinsics, colorIntrinsics, depth2color);
	return alignedColor_devive;
}
