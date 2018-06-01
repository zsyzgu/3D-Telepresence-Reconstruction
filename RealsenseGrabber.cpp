#include "RealsenseGrabber.h"
#include "Parameters.h"
#include "Timer.h"
#include <iostream>

RealsenseGrabber::RealsenseGrabber()
{
	config.enable_stream(RS2_STREAM_DEPTH, DEPTH_W * 2, DEPTH_H * 2, RS2_FORMAT_Z16, 60); // Before decimation filtering W & H are doubled
	config.enable_stream(RS2_STREAM_COLOR, COLOR_W, COLOR_H, RS2_FORMAT_RGBA8, 60);
	profile = pipeline.start(config);
	decimationFilter = new rs2::decimation_filter();
	spatialFilter = new rs2::spatial_filter();
	temporalFilter = new rs2::temporal_filter();
	toDisparityFilter = new rs2::disparity_transform(true);
	toDepthFilter = new rs2::disparity_transform(false);
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
	if (colorData != NULL) {
		delete[] colorData;
	}
}

void RealsenseGrabber::getRGBD(UINT16*& depthData, RGBQUAD*& colorData)
{
	rs2::frameset frame = pipeline.wait_for_frames();
	rs2::frame color = frame.get_color_frame();
	rs2::align align(rs2_stream::RS2_STREAM_COLOR);
	rs2::frameset alignedFrame = align.process(frame);
	rs2::frame depth = alignedFrame.get_depth_frame();

	depth = decimationFilter->process(depth);
	depth = toDisparityFilter->process(depth);
	depth = spatialFilter->process(depth);
	depth = temporalFilter->process(depth);
	depth = toDepthFilter->process(depth);

	depthData = (UINT16*)depth.get_data();
	colorData = (RGBQUAD*)color.get_data();
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
