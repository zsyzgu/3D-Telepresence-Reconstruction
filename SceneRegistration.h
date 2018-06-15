#ifndef REGISTRATION_H
#define REGISTRATION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <Windows.h>
#include "TsdfVolume.cuh"

static const cv::Size boardSize = cv::Size(9, 6);

class SceneRegistration {
private:
	static bool runCalibration(cv::Size imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, std::vector<cv::Point2f> points);
	static cv::Mat worldToPixelMatrix(cv::Mat cameraMatrix, cv::Mat distCoeffs, cv::Mat rotm, cv::Mat tvec);

public:
	static Transformation align(RGBQUAD* source, RGBQUAD* target);
};

#endif
