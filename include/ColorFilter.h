#ifndef COLOR_FILTER_H
#define COLOR_FILTER_H

#include <Windows.h>
#include "Parameters.h"

class ColorFilter
{
	UINT8* data_device;
	RGBQUAD* color_device;
public:
	ColorFilter();
	~ColorFilter();
	void process(int cameraId, UINT8* colorMap);
	RGBQUAD* getCurrFrame_device() {
		return color_device;
	}
};

#endif