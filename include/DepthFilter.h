#ifndef DEPTH_FILTER_H
#define DEPTH_FILTER_H

#include <Windows.h>
#include "Parameters.h"

class DepthFilter {
	/*
	Parameters:
	@depth_device: temp GPU memory of the processing depth images (Z16 disparity).
	@depthFloat_device: GPU memory of the result of depth images (depth in meter).
	@lastFrame_device: the last frame of depth images for temporal filtering.
	Functions:
	@process(cameraId, depthMap): CPU Z16 disparity --> GPU depth in meter (id=cameraId).
	@setConvertFactor(): the convert factor should be set for the transformation from disparity to depth.
	*/
private:
	UINT16* depth_device;
	float* depthFloat_device;
	float* lastFrame_device;
	float convertFactor[MAX_CAMERAS];
public:
	DepthFilter();
	~DepthFilter();
	void process(int cameraId, UINT16* depthMap);
	void setConvertFactor(int cameraId, float converFactor) { this->convertFactor[cameraId] = converFactor; }
	float* getCurrFrame_device() { return depthFloat_device; }
};

#endif
