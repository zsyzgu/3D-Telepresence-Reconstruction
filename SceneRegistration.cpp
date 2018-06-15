#include "SceneRegistration.h"
#include "Timer.h"
#include "Parameters.h"

std::vector<std::vector<cv::Point2f> > packVectors(std::vector<cv::Point2f> pts) {
	std::vector<std::vector<cv::Point2f> > result(1);
	result[0] = pts;
	return result;
}

std::vector<std::vector<cv::Point3f> > packVectors(std::vector<cv::Point3f> pts) {
	std::vector<std::vector<cv::Point3f> > result(1);
	result[0] = pts;
	return result;
}

Transformation SceneRegistration::align(RGBQUAD* source, RGBQUAD* target)
{
	cv::Mat sourceColorMat(COLOR_H, COLOR_W, CV_8UC3);
	cv::Mat targetColorMat(COLOR_H, COLOR_W, CV_8UC3);
	for (int i = 0; i < COLOR_H; i++) {
		for (int j = 0; j < COLOR_W; j++) {
			RGBQUAD color = source[i * COLOR_W + j];
			sourceColorMat.at<cv::Vec3b>(i, j) = cv::Vec3b(color.rgbRed, color.rgbGreen, color.rgbBlue);
			color = target[i * COLOR_W + j];
			targetColorMat.at<cv::Vec3b>(i, j) = cv::Vec3b(color.rgbRed, color.rgbGreen, color.rgbBlue);

		}
	}

	std::vector<cv::Point2f> sourcePoints;
	if (!findChessboardCorners(sourceColorMat, boardSize, sourcePoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE)) {
		puts("Error: Chessboard can't be found in source image.");
		return Transformation();
	}

	std::vector<cv::Point2f> targetPoints;
	if (!findChessboardCorners(targetColorMat, boardSize, targetPoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE)) {
		puts("Error: Chessboard can't be found in target image.");
		return Transformation();
	}

	std::vector<cv::Point3f> objectPoints;
	for (int r = 0; r < boardSize.height; r++) {
		for (int c = 0; c < boardSize.width; c++) {
			objectPoints.push_back(cv::Point3f(c * 0.03f, r * 0.03f, 0));
		}
	}

	cv::Mat sourceCameraMatrix = cv::initCameraMatrix2D(packVectors(objectPoints), packVectors(sourcePoints), sourceColorMat.size(), 0);
	cv::Mat targetCameraMatrix = cv::initCameraMatrix2D(packVectors(objectPoints), packVectors(targetPoints), targetColorMat.size(), 0);
	cv::Mat sourceDistCoeffs;
	cv::Mat targetDistCoeffs;

	cv::Mat rotation, translation, essential, fundamental;
	double rms = cv::stereoCalibrate(
		packVectors(objectPoints),
		packVectors(sourcePoints),
		packVectors(targetPoints),
		sourceCameraMatrix,
		sourceDistCoeffs,
		targetCameraMatrix,
		targetDistCoeffs,
		sourceColorMat.size(),
		rotation,
		translation,
		essential,
		fundamental,
		cv::CALIB_FIX_ASPECT_RATIO +
		cv::CALIB_ZERO_TANGENT_DIST +
		cv::CALIB_USE_INTRINSIC_GUESS +
		cv::CALIB_SAME_FOCAL_LENGTH +
		cv::CALIB_RATIONAL_MODEL +
		cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5
	);

	std::cout << rms << std::endl;

	return Transformation((double*)rotation.data, (double*)translation.data);
}
