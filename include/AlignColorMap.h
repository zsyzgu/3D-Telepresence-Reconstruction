#ifndef ALIGNED_COLOR_MAP_H
#define ALIGNED_COLOR_MAP_H

#include <Windows.h>
#include "Parameters.h"
#include "TsdfVolume.cuh"

class AlignColorMap {
	RGBQUAD* alignedColor_devive;
	bool isRemoveBackground;
	float* depthBackground_device;
	RGBQUAD* colorBackground_device;
public:
	AlignColorMap();
	~AlignColorMap();
	void alignColor2Depth(int cameras, bool* check, float* depth_device, RGBQUAD* color_device, Intrinsics * depthIntrinsics, Intrinsics* colorIntrinsics, Transformation* depth2color);
	bool isBackgroundOn() { return this->isRemoveBackground; };
	void enableBackground();
	void enableBackground(float* depth_device);
	void disableBackground();
	void copyBackground_host2device(float* depthBackground, RGBQUAD* colorBackground);
	void copyBackground_device2host(float* depthBackground, RGBQUAD* colorBackground);
	RGBQUAD* getAlignedColor_device() { return alignedColor_devive; }
};

#endif
