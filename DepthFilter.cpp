#include "DepthFilter.h"
#include "Parameters.h"

extern "C" void cudaDepthFiltering(UINT16* depthMap, UINT16* depth_device, float* depthFloat_device, float* lastFrame_device, float convertFactor);
extern "C" void cudaFilterInit(UINT16*& depth_device, float*& depthFloat_device, float*& lastFrame_device);
extern "C" void cudaFilterClean(UINT16*& depth_device, float*& depthFloat_device, float*& lastFrame_device);

DepthFilter::DepthFilter()
{
	cudaFilterInit(depth_device, depthFloat_device, lastFrame_device);
}

DepthFilter::~DepthFilter()
{
	cudaFilterClean(depth_device, depthFloat_device, lastFrame_device);
}

void DepthFilter::process(int cameraId, UINT16* depthMap) {
	cudaDepthFiltering(depthMap, depth_device, depthFloat_device + cameraId * DEPTH_H * DEPTH_W, lastFrame_device + cameraId * DEPTH_H * DEPTH_W, convertFactor[cameraId]);
}
