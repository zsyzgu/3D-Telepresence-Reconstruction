#ifndef REALSENSE_GRABBER_H
#define REALSENSE_GRABBER_H

#include <Windows.h>
#include <librealsense2/rs.hpp>

class RealsenseGrabber {
private:
	rs2::config config;
	rs2::pipeline pipeline;
	rs2::pipeline_profile profile;

	rs2::decimation_filter* decimationFilter;
	rs2::spatial_filter* spatialFilter;
	rs2::temporal_filter* temporalFilter;
	rs2::disparity_transform* toDisparityFilter;
	rs2::disparity_transform* toDepthFilter;

	RGBQUAD* colorData;

public:
	RealsenseGrabber();
	~RealsenseGrabber();
	void getRGBD(UINT16*& depthData, RGBQUAD*& colorData);
	void showIntrinsics();
};

#endif
