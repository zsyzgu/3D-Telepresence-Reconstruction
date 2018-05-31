#include "RealsenseGrabber.h"
#include "Parameters.h"
#include <iostream>

RealsenseGrabber::RealsenseGrabber()
{
	config.enable_stream(RS2_STREAM_DEPTH, DEPTH_W * 2, DEPTH_H * 2, RS2_FORMAT_Z16, 90); // Before decimation filtering W & H are doubled
	config.enable_stream(RS2_STREAM_COLOR, COLOR_W, COLOR_H, RS2_FORMAT_RGB8, 30);
	profile = pipeline.start(config);
	decimationFilter = new rs2::decimation_filter();
	spatialFilter = new rs2::spatial_filter();
	temporalFilter = new rs2::temporal_filter();
	toDisparityFilter = new rs2::disparity_transform(true);
	toDepthFilter = new rs2::disparity_transform(false);
	depthData = new UINT16[DEPTH_H * DEPTH_W];
	colorData = new RGBQUAD[COLOR_H * COLOR_W];
}

RealsenseGrabber::~RealsenseGrabber()
{
	if (decimationFilter != NULL) {
		delete decimationFilter;
	}
	if (spatialFilter != NULL) {
		delete spatialFilter;
	}
	if (temporalFilter != NULL) {
		delete temporalFilter;
	}
	if (toDisparityFilter != NULL) {
		delete toDisparityFilter;
	}
	if (toDepthFilter != NULL) {
		delete toDepthFilter;
	}
	if (depthData != NULL) {
		delete[] depthData;
	}
	if (colorData != NULL) {
		delete[] colorData;
	}
}

void RealsenseGrabber::getRGBD(UINT16*& depthData, RGBQUAD*& colorData)
{
	rs2::frameset data = pipeline.wait_for_frames();
	rs2::frame depth = data.get_depth_frame();
	rs2::frame color = data.get_color_frame();

	depth = decimationFilter->process(depth);
	depth = toDisparityFilter->process(depth);
	depth = spatialFilter->process(depth);
	depth = temporalFilter->process(depth);
	depth = toDepthFilter->process(depth);

	memcpy(this->depthData, depth.get_data(), DEPTH_H * DEPTH_W * sizeof(UINT16));
	UINT8* colorBuffer = (UINT8*)color.get_data();
#pragma openmp parallel for 
	for (int r = 0; r < COLOR_H; r++) {
		int id = r * COLOR_W;
		for (int c = 0; c < COLOR_W; c++, id++) {
			this->colorData[id].rgbRed = colorBuffer[id * 3 + 0];
			this->colorData[id].rgbGreen = colorBuffer[id * 3 + 1];
			this->colorData[id].rgbBlue = colorBuffer[id * 3 + 2];
		}
	}

	depthData = this->depthData;
	colorData = this->colorData;
}

void RealsenseGrabber::showIntrinsics()
{
	rs2::video_stream_profile depthStream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
	rs2_intrinsics depthIntrinscs = depthStream.get_intrinsics();
	std::cout << "Depth Intrinscs:" << std::endl;
	std::cout << "Fx = " << depthIntrinscs.fx << ", Fy = " << depthIntrinscs.fy << std::endl;
	std::cout << "Cx = " << depthIntrinscs.ppx << ", Cy = " << depthIntrinscs.ppy << std::endl;

	rs2::video_stream_profile colorStream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
	rs2_intrinsics colorIntrinscs = colorStream.get_intrinsics();
	std::cout << "Color Intrinscs:" << std::endl;
	std::cout << "Fx = " << colorIntrinscs.fx << ", Fy = " << colorIntrinscs.fy << std::endl;
	std::cout << "Cx = " << colorIntrinscs.ppx << ", Cy = " << colorIntrinscs.ppy << std::endl;
}
