#include "SceneRegistration.h"
#include "Timer.h"
#include "Parameters.h"

std::vector<std::vector<float> > SceneRegistration::getDepth(RealsenseGrabber* grabber) {
	const cv::Size BOARD_SIZE = cv::Size(9, 6);
	const int BOARD_NUM = BOARD_SIZE.width * BOARD_SIZE.height;
	const float GRID_SIZE = 0.028f;
	const int ITERATION = 10;

	std::vector<cv::Point3f> objectPoints;
	for (int r = 0; r < BOARD_SIZE.height; r++) {
		for (int c = 0; c < BOARD_SIZE.width; c++) {
			objectPoints.push_back(cv::Point3f(c * GRID_SIZE, r * GRID_SIZE, 0));
		}
	}

	Intrinsics* colorIntrinsics;
	RGBQUAD** colorImages;
	std::vector<cv::Point2f> sourcePoints;
	cv::Mat sourceColorMat(COLOR_H, COLOR_W, CV_8UC3);

	std::vector<std::vector<float> > depths;
	int cameras = grabber->getRGB(colorImages, colorIntrinsics);
	for (int id = 0; id < cameras; id++) {
		std::vector<std::vector<cv::Point2f> > sourcePointsArray;
		for (int iter = 0; iter < ITERATION;) {

			RGBQUAD* source = colorImages[id];
			for (int i = 0; i < COLOR_H; i++) {
				for (int j = 0; j < COLOR_W; j++) {
					RGBQUAD color;
					color = source[i * COLOR_W + j];
					sourceColorMat.at<cv::Vec3b>(i, j) = cv::Vec3b(color.rgbRed, color.rgbGreen, color.rgbBlue);
				}
			}
			sourcePoints.clear();
			findChessboardCorners(sourceColorMat, BOARD_SIZE, sourcePoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

			cv::Scalar color = cv::Scalar(0, 0, 255);
			if (sourcePoints.size() == BOARD_NUM) {
				color = cv::Scalar(0, 255, 255);
			}
			for (int i = 0; i < sourcePoints.size(); i++) {
				cv::circle(sourceColorMat, sourcePoints[i], 3, color, 2);
			}
			cv::imshow("Get Depth", sourceColorMat);

			char ch = cv::waitKey(1);
			if (int(ch) != -1) {
				iter = 0;
			}

			if (iter != -1 && sourcePoints.size() == BOARD_NUM) {
				iter++;
				sourcePointsArray.push_back(sourcePoints);
			}
		}

		std::vector<std::vector<cv::Point3f> > objectPointsArray;
		for (int i = 0; i < sourcePointsArray.size(); i++) {
			objectPointsArray.push_back(objectPoints);
		}

		cv::Mat cameraMatrix, distCoeffs;
		std::vector<cv::Mat> rvec, tvec;
		std::vector<float> reprojError;

		double rms = calibrateCamera(objectPointsArray,
			sourcePointsArray,
			sourceColorMat.size(),
			cameraMatrix,
			distCoeffs,
			rvec,
			tvec,
			CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);

		if (!(cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs))) {
			std::cout << "Calibration failed\n";
		}

		cv::Mat rv(3, 1, CV_64FC1);
		cv::Mat tv(3, 1, CV_64FC1);

		cv::Mat sourceCameraMatrix(cv::Size(3, 3), CV_32F);
		sourceCameraMatrix.at<float>(0, 0) = colorIntrinsics[0].fx;
		sourceCameraMatrix.at<float>(1, 1) = colorIntrinsics[0].fy;
		sourceCameraMatrix.at<float>(0, 2) = colorIntrinsics[0].ppx;
		sourceCameraMatrix.at<float>(1, 2) = colorIntrinsics[0].ppy;

		solvePnP(objectPointsArray[0], sourcePointsArray[0], sourceCameraMatrix, distCoeffs, rv, tv);

		std::vector<float> depth;
		for (int i = 0; i < objectPoints.size(); i++)
			depth.push_back(cv::norm(objectPoints[i] - cv::Point3f(tv)));
		depths.push_back(depth);
	}
	return depths;
}


void SceneRegistration::align(RealsenseGrabber* grabber, Transformation* colorTrans)
{
	const cv::Size BOARD_SIZE = cv::Size(9, 6);
	const int BOARD_NUM = BOARD_SIZE.width * BOARD_SIZE.height;
	const float GRID_SIZE = 0.028f;
	const int ITERATION = 10;
	const int CORNERS[4] = { 0, 8, 53, 45 };
	const int RECT_DIST_THRESHOLD = 50;
	const int RECT_AREA_THRESHOLD = 20000;

	RGBQUAD** colorImages;
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
		std::vector<cv::Point2f> rects;
		for (int iter = 0; iter < ITERATION;) {
			cameras = grabber->getRGB(colorImages, colorIntrinsics);
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
			cv::Scalar color = cv::Scalar(0, 0, 255);
			if (sourcePoints.size() == BOARD_NUM) {
				color = cv::Scalar(0, 255, 0);
			}
			for (int i = 0; i < sourcePoints.size(); i++) {
				cv::circle(sourceColorMat, sourcePoints[i], 3, color, 2);
			}
			for (int i = 0; i < targetPoints.size(); i++) {
				cv::circle(targetColorMat, targetPoints[i], 3, color, 2);
			}
			cv::Mat mergeImage;
			bool valid = (iter != -1 && sourcePoints.size() == BOARD_NUM && targetPoints.size() == BOARD_NUM);
			
			for (int i = 0; i < rects.size(); i += 4) {
				for (int j = 0; j < 4; j++) {
					cv::line(sourceColorMat, rects[i + j], rects[i + (j + 1) % 4], cv::Scalar(0, 255, 0), 2);
				}
			}
			if (valid) {
				cv::Point2f p0 = sourcePoints[CORNERS[0]];
				cv::Point2f p1 = sourcePoints[CORNERS[1]];
				cv::Point2f p2 = sourcePoints[CORNERS[2]];
				cv::Point2f p3 = sourcePoints[CORNERS[3]];
				float area = (cv::norm(p0 - p1) + cv::norm(p2 - p3)) * (cv::norm(p0 - p3) + cv::norm(p1 - p2)) / 4;
				std::cout << area << std::endl;
				if (area < RECT_AREA_THRESHOLD) {
					valid = false;
				}
				for (int i = 0; i < rects.size() / 4; i++) {
					float dist = (cv::norm(rects[i * 4 + 0] - p0)
						+ cv::norm(rects[i * 4 + 1] - p1)
						+ cv::norm(rects[i * 4 + 2] - p2)
						+ cv::norm(rects[i * 4 + 3] - p3)) / 4;
					if (dist < RECT_DIST_THRESHOLD) {
						valid = false;
					}
				}
				cv::Scalar color = valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
				for (int i = 0; i < 4; i++) {
					cv::line(sourceColorMat, sourcePoints[CORNERS[i]], sourcePoints[CORNERS[(i + 1) % 4]], color, 2);
				}
			}

			cv::hconcat(sourceColorMat, targetColorMat, mergeImage);
			cv::imshow("Calibration", mergeImage);

			char ch = cv::waitKey(1);
			if (int(ch) != -1) {
				iter = 0;
			}

			if (valid) {
				iter++;
				sourcePointsArray.push_back(sourcePoints);
				targetPointsArray.push_back(targetPoints);
				for (int i = 0; i < 4; i++) {
					rects.push_back(sourcePoints[CORNERS[i]]);
				}
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
