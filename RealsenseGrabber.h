#ifndef REALSENSE_GRABBER_H
#define REALSENSE_GRABBER_H

#include <librealsense2/rs.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <map>
#include <Windows.h>
#include <pcl/point_cloud.h>
#include "Parameters.h"
#include "TsdfVolume.cuh"

class RealsenseGrabber
{
private:
	std::mutex _mutex;

	struct ViewPort {
		rs2::pipeline pipe;
		rs2::pipeline_profile profile;
	};
	std::map<std::string, rs2::pipeline> devices;

	rs2::decimation_filter* decimationFilter[MAX_CAMERAS];
	rs2::spatial_filter* spatialFilter[MAX_CAMERAS];
	rs2::temporal_filter* temporalFilter[MAX_CAMERAS];
	rs2::disparity_transform* toDisparityFilter[MAX_CAMERAS];
	rs2::disparity_transform* toDepthFilter[MAX_CAMERAS];

	void enableDevice(rs2::device device);

	UINT16** depthImages;
	RGBQUAD** colorImages;
	Transformation* colorTrans;
	Intrinsics* depthIntrinsics;
	Intrinsics* colorIntrinsics;

public:
	RealsenseGrabber();
	~RealsenseGrabber();
	int getRGBD(UINT16**& depthImages, RGBQUAD**& colorImages, Transformation*& colorTrans, Intrinsics*& depthIntrinsics, Intrinsics*& colorIntrinsics);
};

#endif
