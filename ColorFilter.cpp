#include <iostream>
#include "ColorFilter.h"

extern "C" void cudaColorFiltering(UINT8* colorMap, UINT8* source_device, RGBQUAD* color_device);
extern "C" void cudaColorFilterInit(UINT8*& source_device, RGBQUAD*& color_device);
extern "C" void cudaColorFilterClean(UINT8*& source_device, RGBQUAD*& color_device);

ColorFilter::ColorFilter()
{
	cudaColorFilterInit(data_device, color_device);
}

ColorFilter::~ColorFilter()
{
	cudaColorFilterClean(data_device, color_device);
}

void ColorFilter::process(int cameraId, UINT8* colorMap)
{
	cudaColorFiltering(colorMap, data_device, color_device + cameraId * COLOR_H * COLOR_W);
}
