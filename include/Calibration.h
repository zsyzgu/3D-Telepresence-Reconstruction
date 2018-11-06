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
	Consts:
	@INTERATION: The number of captured images in the calibration.
	@RECT_DIST_THRESHOLD: In the calibration, the captured checkerboard should be big enough.
	@RECT_AREA_THRESHOLD: In the calibration, the captured checkerboards should be far away enough from each others.
	@(others): Consts of the checkerboard.
	Parameters:
	@world2color: The extrinsics from the world coordinate to the color cameras.
	@world2depth: The extrinsics from the world coordinate to the depth cameras.
	@checkerboardPoints: The points cloud of the checkerboard when it is at the origin point.
	Functions:
	@setOrigin(): Set the position of the checkerboard as the coordinate origin.
	@align(): Align all the color cameras - calcuate @world2color
	Notes:
	In the calibration, we use the color images as the cues. So the output is the extrinsics of color cameras.
	Next, we get the extrinsics between color camera and depthe cameras (color2depth) from the Realsense SDK.
	Then, world2depth = color2depth * world2color.
	Last, we used world2depth in the TSDF Volume.
	*/
private:
	const int ITERATION = 5;
	const int RECT_DIST_THRESHOLD = COLOR_H / 20;
	const int RECT_AREA_THRESHOLD = COLOR_H * COLOR_W / 100;
	const cv::Size BOARD_SIZE = cv::Size(9, 6);
	const int BOARD_NUM = BOARD_SIZE.width * BOARD_SIZE.height;
	const float GRID_SIZE = 0.02513f;
	const int CORNERS[4] = { 0, 8, 53, 45 };
private:
	Transformation* world2color;
	Transformation* world2depth;
	std::vector<cv::Point3f> checkerboardPoints;
	void initCheckerboardPoints();
	Transformation calnInv(Transformation T);
	void rgb2mat(cv::Mat* mat, RGBQUAD* rgb);
	cv::Mat intrinsics2mat(Intrinsics T);
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
