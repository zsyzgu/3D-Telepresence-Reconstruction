#include "DepthFilter.h"
#include "Parameters.h"

extern "C" void cudaDepthFiltering(UINT16* depthMap, UINT16* depth_device, float* depthFloat_device, float* lastFrame_device, float convertFactor);
extern "C" void cudaDepthFilterInit(UINT16*& depth_device, float*& depthFloat_device, float*& lastFrame_device);
extern "C" void cudaDepthFilterClean(UINT16*& depth_device, float*& depthFloat_device, float*& lastFrame_device);

DepthFilter::DepthFilter()
{
	cudaDepthFilterInit(depth_device, depthFloat_device, lastFrame_device);
}

DepthFilter::~DepthFilter()
{
	cudaDepthFilterClean(depth_device, depthFloat_device, lastFrame_device);
}

void DepthFilter::process(int cameraId, UINT16* depthMap)
{
	cudaDepthFiltering(depthMap, depth_device, depthFloat_device + cameraId * DEPTH_H * DEPTH_W, lastFrame_device + cameraId * DEPTH_H * DEPTH_W, convertFactor[cameraId]);
}
