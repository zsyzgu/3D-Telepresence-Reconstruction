#include "SceneRegistration.h"
#include "Timer.h"
#include "Parameters.h"

Transformation SceneRegistration::align(RealsenseGrabber* grabber)
{
	const cv::Size boardSize = cv::Size(9, 6);
	const float gridSize = 0.028f;
	const int ITERATION = 3;

	UINT16** depthImages;
	RGBQUAD** colorImages;
	Transformation* depthTrans;
	Intrinsics* depthIntrinsics;
	Intrinsics* colorIntrinsics;

	std::vector<std::vector<cv::Point2f> > sourcePointsArray;
	std::vector<std::vector<cv::Point2f> > targetPointsArray;
	std::vector<std::vector<cv::Point3f> > objectPointsArray;
	std::vector<cv::Point2f> sourcePoints;
	std::vector<cv::Point2f> targetPoints;
	cv::Mat sourceColorMat(COLOR_H, COLOR_W, CV_8UC3);
	cv::Mat targetColorMat(COLOR_H, COLOR_W, CV_8UC3);

	std::vector<cv::Point3f> objectPoints;
	for (int r = 0; r < boardSize.height; r++) {
		for (int c = 0; c < boardSize.width; c++) {
			objectPoints.push_back(cv::Point3f(c * gridSize, r * gridSize, 0));
		}
	}

	for (int iter = 0; iter < ITERATION; iter++) {
		std::cout << "Interation = " << iter << " / " << ITERATION << std::endl;

		while (true) {
			grabber->getRGBD(depthImages, colorImages, depthTrans, depthIntrinsics, colorIntrinsics);
			RGBQUAD* source = colorImages[0];
			RGBQUAD* target = colorImages[1];
			for (int i = 0; i < COLOR_H; i++) {
				for (int j = 0; j < COLOR_W; j++) {
					RGBQUAD color;
					color = source[i * COLOR_W + j];
					sourceColorMat.at<cv::Vec3b>(i, j) = cv::Vec3b(color.rgbRed, color.rgbGreen, color.rgbBlue);
					color = target[i * COLOR_W + j];
					targetColorMat.at<cv::Vec3b>(i, j) = cv::Vec3b(color.rgbRed, color.rgbGreen, color.rgbBlue);
				}
			}
			cv::Mat mergeImage;
			cv::hconcat(sourceColorMat, targetColorMat, mergeImage);

			sourcePoints.clear();
			targetPoints.clear();
			findChessboardCorners(sourceColorMat, boardSize, sourcePoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
			findChessboardCorners(targetColorMat, boardSize, targetPoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
			for (int i = 0; i < sourcePoints.size(); i++) {
				cv::circle(mergeImage, sourcePoints[i], 3, cv::Scalar(0, 0, 255), 2);
			}
			for (int i = 0; i < targetPoints.size(); i++) {
				cv::Point2f pos = targetPoints[i] + cv::Point2f(sourceColorMat.cols, 0);
				cv::circle(mergeImage, pos, 3, cv::Scalar(0, 0, 255), 2);
				if (i < sourcePoints.size()) {
					cv::line(mergeImage, sourcePoints[i], pos, cv::Scalar(0, 255, 255));
				}
			}

			cv::imshow("Calibration", mergeImage);
			char ch = cv::waitKey(1);
			if (int(ch) != -1) {
				break;
			}
		}
		sourcePointsArray.push_back(sourcePoints);
		targetPointsArray.push_back(targetPoints);
		objectPointsArray.push_back(objectPoints);
	}
	cv::destroyAllWindows();

	//cv::Mat sourceCameraMatrix = cv::initCameraMatrix2D(objectPointsArray, sourcePointsArray, sourceColorMat.size(), 0);
	//cv::Mat targetCameraMatrix = cv::initCameraMatrix2D(objectPointsArray, targetPointsArray, targetColorMat.size(), 0);
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
		fundamental
	);

	return Transformation((double*)rotation.data, (double*)translation.data);
}
