#ifndef REALSENSE_GRABBER_H
#define REALSENSE_GRABBER_H

#include <librealsense2/rs.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <map>
#include <Windows.h>
#include "Parameters.h"
#include "TsdfVolume.cuh"
#include "DepthFilter.h"
#include "ColorFilter.h"
#include "AlignColorMap.h"
#include "Transmission.h"

class RealsenseGrabber
{
private:
	std::vector<rs2::pipeline> devices;
	std::vector<float> convertFactors;

	DepthFilter* depthFilter;
	ColorFilter* colorFilter;
	AlignColorMap* alignColorMap;

	void enableDevice(rs2::device device);

	UINT16** depthImages;
	UINT8** colorImages;
	RGBQUAD** colorImagesRGB;
	Transformation* depth2color;
	Transformation* color2depth;
	Intrinsics* depthIntrinsics;
	Intrinsics* colorIntrinsics;
	Intrinsics* originColorIntrinsics;
	Transmission* transmission;

public:
	RealsenseGrabber();
	~RealsenseGrabber();
	int getRGBD(float*& depthImages_device, RGBQUAD*& colorImages_device, Transformation* extrinsics);
	int getRGB(RGBQUAD**& colorImages);
	void saveBackground();
	void loadBackground();
	void setTransmission(Transmission* transmission) { this->transmission = transmission; }
	Intrinsics* getDepthIntrinsics() { return depthIntrinsics; }
	Intrinsics* getColorIntrinsics() { return colorIntrinsics; }
	Intrinsics* getOriginColorIntrinsics() { return originColorIntrinsics; }
	Transformation* getColor2Depth() { return color2depth; }
};

#endif
