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
	struct ViewPort {
		rs2::pipeline pipe;
		rs2::pipeline_profile profile;
	};
	std::vector<rs2::pipeline> devices;
	std::vector<float> convertFactors;

	DepthFilter* depthFilter;
	ColorFilter* colorFilter;
	AlignColorMap* alignColorMap;

	void enableDevice(rs2::device device);
	void convertYUVtoRGBA(UINT8* src, RGBQUAD* dst);

	UINT16** depthImages;
	UINT8** colorImages;
	RGBQUAD** colorImagesRGB;
	Transformation* depth2color;
	Transformation* color2depth;
	Intrinsics* depthIntrinsics;
	Intrinsics* colorIntrinsics;
	Transmission* transmission;

public:
	RealsenseGrabber();
	~RealsenseGrabber();
	int getRGBD(float*& depthImages_device, RGBQUAD*& colorImages_device, Transformation* world2depth, Transformation* world2color, Intrinsics*& depthIntrinscis, Intrinsics*& colorIntrinsics);
	int getRGB(RGBQUAD**& colorImages, Intrinsics*& colorIntrinsics);
	void saveBackground();
	void loadBackground();
	void setTransmission(Transmission* transmission) { this->transmission = transmission; }
};

#endif
