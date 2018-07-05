#include "DepthFilter.h"

extern "C" void cudaDepthFiltering(UINT16* depthMap);

DepthFilter::DepthFilter()
{

}

DepthFilter::~DepthFilter()
{

}

void DepthFilter::process(UINT16* depthMap) {
	cudaDepthFiltering(depthMap);
}
