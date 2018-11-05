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

class RealsenseGrabber
{
	/*
	Parameters:
	@devices: all the opened Realsense cameras.
	@depthFilter: to filter depth images and to store them to GPU memory.
	@colorFilter: to filter color images and to store them to GPU memory.
	@alignColorMap: to map color images to the intrinsics of depth images. Also to store the aligned color images.
	@depthImages: temp CPU memory to fetch data from Realsense.
	@colorImages: temp CPU memory to fetch data from Realsense.
	@colorImagesRGB: -------------- TO REMOVE ------------------
	@depth2color: the extrinsics from depth image to color image of each camera.
	@color2depth: the extrinsics from color image to depth image of each camera.
	@depthIntrinsics: the depth intrinsics of each camera.
	@colorIntrinsics: the color intrinsics of each camera (after the alignment to depth intrinsics).
	@originColorIntrinsics: the color intrinsics of each camera (before the alignment).
	Functions:
	@updateRGBD(): update depth/color/aligned color images of the current frame.
	@saveBackground(): save the background map (if a map was already saved, it will be removal).
	@loadBackground(): load the background map from Background.cfg.
	*/
private:
	std::vector<rs2::pipeline> devices;
	DepthFilter* depthFilter;
	ColorFilter* colorFilter;
	AlignColorMap* alignColorMap;
	UINT16** depthImages;
	UINT8** colorImages;
	RGBQUAD** colorImagesRGB;
	Transformation* depth2color;
	Transformation* color2depth;
	Intrinsics* depthIntrinsics;
	Intrinsics* colorIntrinsics;
	Intrinsics* originColorIntrinsics;

	void enableDevice(rs2::device device);

public:
	RealsenseGrabber();
	~RealsenseGrabber();
	void updateRGBD();
	void getRGB(RGBQUAD**& colorImages);
	void saveBackground();
	void loadBackground();
	int getCameras() { return devices.size(); }
	Intrinsics* getDepthIntrinsics() { return depthIntrinsics; }
	Intrinsics* getColorIntrinsics() { return colorIntrinsics; }
	Intrinsics* getOriginColorIntrinsics() { return originColorIntrinsics; }
	Transformation* getColor2Depth() { return color2depth; }
	float* getDepthImages_device() { return depthFilter->getCurrFrame_device(); }
	RGBQUAD* getColorImages_device() { return alignColorMap->getAlignedColor_device(); }
	RGBQUAD* getOriginColorImages_device() { return colorFilter->getCurrFrame_device(); }
};

#endif
