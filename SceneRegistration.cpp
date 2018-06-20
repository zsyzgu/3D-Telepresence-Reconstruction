#include "SceneRegistration.h"
#include "Timer.h"
#include "Parameters.h"

Transformation SceneRegistration::align(RealsenseGrabber* grabber, Transformation* colorTrans)
{
	const cv::Size BOARD_SIZE = cv::Size(9, 6);
	const int BOARD_NUM = BOARD_SIZE.width * BOARD_SIZE.height;
	const float GRID_SIZE = 0.028f;
	const int ITERATION = 16;

	UINT16** depthImages;
	RGBQUAD** colorImages;
	Transformation* depthTrans;
	Intrinsics* depthIntrinsics;
	Intrinsics* colorIntrinsics;
	std::vector<cv::Point2f> sourcePoints;
	std::vector<cv::Point2f> targetPoints;
	cv::Mat sourceColorMat(COLOR_H, COLOR_W, CV_8UC3);
	cv::Mat targetColorMat(COLOR_H, COLOR_W, CV_8UC3);

	std::vector<cv::Point3f> objectPoints;
	for (int r = 0; r < BOARD_SIZE.height; r++) {
		for (int c = 0; c < BOARD_SIZE.width; c++) {
			objectPoints.push_back(cv::Point3f(c * GRID_SIZE, r * GRID_SIZE, 0));
		}
	}
	
	int cameras = -1;
	for (int targetId = 1; cameras == -1 || targetId < cameras; targetId++) {
		std::vector<std::vector<cv::Point2f> > sourcePointsArray;
		std::vector<std::vector<cv::Point2f> > targetPointsArray;

		for (int iter = -1; iter < ITERATION;) {
			cameras = grabber->getRGBD(depthImages, colorImages, depthTrans, depthIntrinsics, colorIntrinsics);
			RGBQUAD* source = colorImages[0];
			RGBQUAD* target = colorImages[targetId];
			for (int i = 0; i < COLOR_H; i++) {
				for (int j = 0; j < COLOR_W; j++) {
					RGBQUAD color;
					color = source[i * COLOR_W + j];
					sourceColorMat.at<cv::Vec3b>(i, j) = cv::Vec3b(color.rgbRed, color.rgbGreen, color.rgbBlue);
					color = target[i * COLOR_W + j];
					targetColorMat.at<cv::Vec3b>(i, j) = cv::Vec3b(color.rgbRed, color.rgbGreen, color.rgbBlue);
				}
			}

			sourcePoints.clear();
			targetPoints.clear();
			findChessboardCorners(sourceColorMat, BOARD_SIZE, sourcePoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
			findChessboardCorners(targetColorMat, BOARD_SIZE, targetPoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
			for (int i = 0; i < sourcePoints.size(); i++) {
				cv::Scalar color = cv::Scalar(0, 0, 255);
				if (sourcePoints.size() == BOARD_NUM) {
					color = cv::Scalar(0, 255, 0);
				}
				cv::circle(sourceColorMat, sourcePoints[i], 3, color, 2);
			}
			for (int i = 0; i < targetPoints.size(); i++) {
				cv::Scalar color = cv::Scalar(0, 0, 255);
				if (targetPoints.size() == BOARD_NUM) {
					color = cv::Scalar(0, 255, 0);
				}
				cv::circle(targetColorMat, targetPoints[i], 3, color, 2);
			}
			cv::Mat mergeImage;
			cv::hconcat(sourceColorMat, targetColorMat, mergeImage);

			cv::imshow("Calibration", mergeImage);
			char ch = cv::waitKey(1);
			if (int(ch) != -1) {
				iter = 0;
			}

			if (iter != -1 && sourcePoints.size() == BOARD_NUM && targetPoints.size() == BOARD_NUM) {
				iter++;
				sourcePointsArray.push_back(sourcePoints);
				targetPointsArray.push_back(targetPoints);
			}
		}

		std::vector<std::vector<cv::Point3f> > objectPointsArray;
		for (int i = 0; i < sourcePointsArray.size(); i++) {
			objectPointsArray.push_back(objectPoints);
		}

		cv::Mat sourceCameraMatrix(cv::Size(3, 3), CV_32F);
		sourceCameraMatrix.at<float>(0, 0) = colorIntrinsics[0].fx;
		sourceCameraMatrix.at<float>(1, 1) = colorIntrinsics[0].fy;
		sourceCameraMatrix.at<float>(0, 2) = colorIntrinsics[0].ppx;
		sourceCameraMatrix.at<float>(1, 2) = colorIntrinsics[0].ppy;
		sourceCameraMatrix.at<float>(2, 2) = 1;
		cv::Mat targetCameraMatrix(cv::Size(3, 3), CV_32F);
		targetCameraMatrix.at<float>(0, 0) = colorIntrinsics[targetId].fx;
		targetCameraMatrix.at<float>(1, 1) = colorIntrinsics[targetId].fy;
		targetCameraMatrix.at<float>(0, 2) = colorIntrinsics[targetId].ppx;
		targetCameraMatrix.at<float>(1, 2) = colorIntrinsics[targetId].ppy;
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
			fundamental
		);

		colorTrans[targetId] = Transformation((double*)rotation.data, (double*)translation.data);
	}
	cv::destroyAllWindows();
}
