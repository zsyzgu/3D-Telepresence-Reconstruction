#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include "TsdfVolume.cuh"
#include "RealsenseGrabber.h"
#include "Configuration.h"

class Calibration {
	/*
	Parameters:
	@world2color:	The extrinsics from the world coordinate to the color cameras.
	@world2depth:	The extrinsics from the world coordinate to the depth cameras.
	Functions:
	@setOrigin():	Set the position of the checkerboard as the coordinate origin.
	@align():		Align all the color cameras - calcuate @world2color
	Notes:
	In the calibration, we use the color images as the cues. So the output is the extrinsics of color cameras.
	Next, we get the extrinsics between color camera and depthe cameras (color2depth) from the Realsense SDK.
	Then, world2depth = color2depth * world2color.
	Last, we used world2depth in the TSDF Volume.
	*/
private:
	Transformation* world2color;
	Transformation* world2depth;
	void updateWorld2Depth(int cameras, RealsenseGrabber* grabber);
public:
	Calibration();
	~Calibration();
	void setOrigin(RealsenseGrabber* grabber);
	void align(RealsenseGrabber* grabber, int targetId);
	void align(RealsenseGrabber* grabber);
	Transformation* getExtrinsics() { return world2depth; }
};

#endif
