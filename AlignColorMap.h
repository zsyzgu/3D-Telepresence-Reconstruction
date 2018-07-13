#ifndef ALIGNED_COLOR_MAP_H
#define ALIGNED_COLOR_MAP_H

#include <Windows.h>
#include "Parameters.h"
#include "TsdfVolume.cuh"

class AlignColorMap {
	RGBQUAD* alignedColor_devive;
public:
	AlignColorMap();
	~AlignColorMap();
	RGBQUAD* getAlignedColor_device(int cameras, bool* check, float* depth_device, RGBQUAD* color_device, Intrinsics * depthIntrinsics, Intrinsics* colorIntrinsics, Transformation* depth2color);
};

#endif
