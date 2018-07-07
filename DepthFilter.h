#ifndef DEPTH_FILTER_H
#define DEPTH_FILTER_H

#include <iostream>
#include <thread>
#include <mutex>
#include <map>
#include <Windows.h>
#include <vector>
#include "Parameters.h"

class DepthFilter {
	UINT16* depth_device;
	float* depthFloat_device;
	float* lastFrame_device;
	float convertFactor[MAX_CAMERAS];
public:
	DepthFilter();
	~DepthFilter();
	void process(int cameraId, UINT16* depthMap);
	void setConvertFactor(int cameraId, float converFactor) {
		this->convertFactor[cameraId] = converFactor;
	}
	float* getCurrFrame_device() {
		return depthFloat_device;
	}
};

#endif
