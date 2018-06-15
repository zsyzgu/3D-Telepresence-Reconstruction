#include "SceneRegistration.h"
#include "Timer.h"
#include "Parameters.h"

Transformation SceneRegistration::align(RealsenseGrabber* grabber)
{
	const cv::Size boardSize = cv::Size(9, 6);
	const float gridSize = 0.03f;

	UINT16** depthImages;
	RGBQUAD** colorImages;
	Transformation* depthTrans;
	Intrinsics* depthIntrinsics;
	Intrinsics* colorIntrinsics;

	std::vector<std::vector<cv::Point2f> > sourcePointsArray;
	std::vector<std::vector<cv::Point2f> > targetPointsArray;
	std::vector<std::vector<cv::Point3f> > objectPointsArray;

	grabber->getRGBD(depthImages, colorImages, depthTrans, depthIntrinsics, colorIntrinsics);
	RGBQUAD* source = colorImages[0];
	RGBQUAD* target = colorImages[1];
	cv::Mat sourceColorMat(COLOR_H, COLOR_W, CV_8UC3);
	cv::Mat targetColorMat(COLOR_H, COLOR_W, CV_8UC3);
	for (int i = 0; i < COLOR_H; i++) {
		for (int j = 0; j < COLOR_W; j++) {
			RGBQUAD color;
			color = source[i * COLOR_W + j];
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
	sourcePointsArray.push_back(sourcePoints);

	std::vector<cv::Point2f> targetPoints;
	if (!findChessboardCorners(targetColorMat, boardSize, targetPoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE)) {
		puts("Error: Chessboard can't be found in target image.");
		return Transformation();
	}
	targetPointsArray.push_back(targetPoints);

	std::vector<cv::Point3f> objectPoints;
	for (int r = 0; r < boardSize.height; r++) {
		for (int c = 0; c < boardSize.width; c++) {
			objectPoints.push_back(cv::Point3f(c * gridSize, r * gridSize, 0));
		}
	}
	objectPointsArray.push_back(objectPoints);

	//cv::Mat sourceCameraMatrix = cv::initCameraMatrix2D(packVectors(objectPoints), packVectors(sourcePoints), sourceColorMat.size(), 0);
	//cv::Mat targetCameraMatrix = cv::initCameraMatrix2D(packVectors(objectPoints), packVectors(targetPoints), targetColorMat.size(), 0);
	cv::Mat sourceCameraMatrix(cv::Size(3, 3), CV_32F);
	sourceCameraMatrix.at<float>(0, 0) = colorIntrinsics[0].fx;
	sourceCameraMatrix.at<float>(1, 1) = colorIntrinsics[0].fy;
	sourceCameraMatrix.at<float>(0, 2) = colorIntrinsics[0].ppx;
	sourceCameraMatrix.at<float>(1, 2) = colorIntrinsics[0].ppy;
	sourceCameraMatrix.at<float>(2, 2) = 1;
	cv::Mat targetCameraMatrix(cv::Size(3, 3), CV_32F);
	targetCameraMatrix.at<float>(0, 0) = colorIntrinsics[1].fx;
	targetCameraMatrix.at<float>(1, 1) = colorIntrinsics[1].fy;
	targetCameraMatrix.at<float>(0, 2) = colorIntrinsics[1].ppx;
	targetCameraMatrix.at<float>(1, 2) = colorIntrinsics[1].ppy;
	targetCameraMatrix.at<float>(2, 2) = 1;

	cv::Mat sourceDistCoeffs;
	cv::Mat targetDistCoeffs;
	cv::Mat rotation, translation, essential, fundamental;
	double rms = cv::stereoCalibrate(
		objectPointsArray,
		sourcePointsArray,
		targetPointsArray,
		sourceCameraMatrix,
		sourceDistCoeffs,
		targetCameraMatrix,
		targetDistCoeffs,
		cv::Size(COLOR_H, COLOR_W),
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

	return Transformation((double*)rotation.data, (double*)translation.data);
}
