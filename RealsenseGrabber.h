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

class RealsenseGrabber
{
private:
	struct ViewPort {
		rs2::pipeline pipe;
		rs2::pipeline_profile profile;
	};
	std::vector<rs2::pipeline> devices;
	std::vector<float> convertFactors;

	rs2::decimation_filter* decimationFilter[MAX_CAMERAS];
	DepthFilter* depthFilter;

	void enableDevice(rs2::device device);

	UINT16** depthImages;
	RGBQUAD** colorImages;
	Transformation* depthTrans;
	Intrinsics* depthIntrinsics;
	Intrinsics* colorIntrinsics;

public:
	RealsenseGrabber();
	~RealsenseGrabber();
	int getRGBD(float*& depthImages_device, RGBQUAD**& colorImages, Transformation*& depthTrans, Intrinsics*& depthIntrinsics, Intrinsics*& colorIntrinsics);
	int getRGB(RGBQUAD**& colorImages, Intrinsics*& colorIntrinsics);
};

#endif
