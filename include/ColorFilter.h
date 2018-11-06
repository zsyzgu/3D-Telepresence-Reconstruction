#ifndef COLOR_FILTER_H
#define COLOR_FILTER_H

#include <Windows.h>
#include "Parameters.h"

class ColorFilter
{
	/*
	Parameters:
	@data_device: temp GPU memory of the processing YUV image.
	@color_device: GPU memory of all the RGB images.
	Functions:
	@process(cameraId, colorMap): CPU YUV --> GPU RGB (id=cameraId). 
	*/
private:
	UINT8* data_device;
	RGBQUAD* color_device;
public:
	ColorFilter();
	~ColorFilter();
	void process(int cameraId, UINT8* colorMap);
	RGBQUAD* getCurrFrame_device() { return color_device; }
};

#endif