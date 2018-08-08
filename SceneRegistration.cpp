#include "SceneRegistration.h"
#include "Timer.h"
#include "Parameters.h"

void SceneRegistration::setOrigin(int cameras, RealsenseGrabber* grabber, Transformation* world2color) {
	const cv::Size BOARD_SIZE = cv::Size(9, 6);
	const int BOARD_NUM = BOARD_SIZE.width * BOARD_SIZE.height;
	const float GRID_SIZE = 0.02513f;

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

	int mainId = 0;
	do {
		grabber->getRGB(colorImages, colorIntrinsics);
		RGBQUAD* source = colorImages[mainId];
		for (int i = 0; i < COLOR_H; i++) {
			for (int j = 0; j < COLOR_W; j++) {
				RGBQUAD color = source[i * COLOR_W + j];
				sourceColorMat.at<cv::Vec3b>(i, j) = cv::Vec3b(color.rgbRed, color.rgbGreen, color.rgbBlue);
			}
		}
		sourcePoints.clear();
		findChessboardCorners(sourceColorMat, BOARD_SIZE, sourcePoints, /*cv::CALIB_CB_ADAPTIVE_THRESH | */cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

		cv::Scalar color = cv::Scalar(0, 0, 255);
		if (sourcePoints.size() == BOARD_NUM) {
			color = cv::Scalar(0, 255, 255);
		}
		for (int i = 0; i < sourcePoints.size(); i++) {
			if (i == 0) {
				cv::circle(sourceColorMat, sourcePoints[i], 3, cv::Scalar(255, 0, 0), 2);
			} else {
				cv::circle(sourceColorMat, sourcePoints[i], 3, color, 2);
			}
		}
		cv::imshow("Get Depth", sourceColorMat);
		char ch = cv::waitKey(1);
		if ('0' <= ch && ch < '4') {
			mainId = ch - '0';
		}

	} while (sourcePoints.size() != BOARD_NUM);

	cv::Mat sourceCameraMatrix(cv::Size(3, 3), CV_32F);
	sourceCameraMatrix.at<float>(0, 0) = colorIntrinsics[mainId].fx;
	sourceCameraMatrix.at<float>(1, 1) = colorIntrinsics[mainId].fy;
	sourceCameraMatrix.at<float>(0, 2) = colorIntrinsics[mainId].ppx;
	sourceCameraMatrix.at<float>(1, 2) = colorIntrinsics[mainId].ppy;

	cv::Mat distCoeffs;
	cv::Mat rv(3, 1, CV_64FC1);
	cv::Mat tv(3, 1, CV_64FC1);
	solvePnP(objectPoints, sourcePoints, sourceCameraMatrix, distCoeffs, rv, tv);
	cv::Rodrigues(rv, rv);
	Transformation world2camera((double*)rv.data, (double*)tv.data);

	//Caln Inv of Main Camera
	Transformation* mainTrans = &world2color[mainId];
	double* rvData = (double*)rv.data;
	rvData[0] = mainTrans->rotation0.x, rvData[1] = mainTrans->rotation0.y, rvData[2] = mainTrans->rotation0.z;
	rvData[3] = mainTrans->rotation1.x, rvData[4] = mainTrans->rotation1.y, rvData[5] = mainTrans->rotation1.z;
	rvData[6] = mainTrans->rotation2.x, rvData[7] = mainTrans->rotation2.y, rvData[8] = mainTrans->rotation2.z;
	double* tvData = (double*)tv.data;
	tvData[0] = mainTrans->translation.x, tvData[1] = mainTrans->translation.y, tvData[2] = mainTrans->translation.z;
	rv = rv.inv();
	tv = -rv * tv;
	Transformation camera0Inv = Transformation((double*)rv.data, (double*)tv.data);

	for (int i = 0; i < cameras; i++) {
		world2color[i] = (world2color[i] * camera0Inv) * world2camera;
	}

	cv::destroyAllWindows();
}


void SceneRegistration::align(int cameras, RealsenseGrabber* grabber, Transformation* world2color, int targetId)
{
	if (targetId <= 0 || targetId >= cameras) {
		return;
	}

	const cv::Size BOARD_SIZE = cv::Size(9, 6);
	const int BOARD_NUM = BOARD_SIZE.width * BOARD_SIZE.height;
	const float GRID_SIZE = 0.02513f;
#if CALIBRATION == true
	const int ITERATION = 10;
#else
	const int ITERATION = 1;
#endif
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

	std::vector<std::vector<cv::Point2f> > sourcePointsArray;
	std::vector<std::vector<cv::Point2f> > targetPointsArray;
	std::vector<cv::Point2f> rects;
	for (int iter = 0; iter < ITERATION;) {
		grabber->getRGB(colorImages, colorIntrinsics);
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
		findChessboardCorners(sourceColorMat, BOARD_SIZE, sourcePoints, /*cv::CALIB_CB_ADAPTIVE_THRESH | */cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
		findChessboardCorners(targetColorMat, BOARD_SIZE, targetPoints, /*cv::CALIB_CB_ADAPTIVE_THRESH | */cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
		cv::Scalar color = cv::Scalar(0, 0, 255);
		if (sourcePoints.size() == BOARD_NUM) {
			color = cv::Scalar(0, 255, 0);
		}
		for (int i = 0; i < sourcePoints.size(); i++) {
			if (i == 0) {
				cv::circle(sourceColorMat, sourcePoints[i], 3, cv::Scalar(255, 0, 0), 2);
			} else {
				cv::circle(sourceColorMat, sourcePoints[i], 3, color, 2);
			}
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
#if CALIBRATION == true
		cv::pyrDown(mergeImage, mergeImage, cv::Size(mergeImage.cols / 2, mergeImage.rows / 2));
#endif
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
	std::cout << "RMS [" << targetId << "]= " << rms << std::endl;

	world2color[targetId] = Transformation((double*)rotation.data, (double*)translation.data);
	cv::destroyAllWindows();
}

void SceneRegistration::align(int cameras, RealsenseGrabber* grabber, Transformation* world2color)
{
	world2color[0].setIdentity();
	for (int targetId = 1; targetId < cameras; targetId++) {
		align(cameras, grabber, world2color, targetId);
	}
}

void SceneRegistration::adjust(int cameras, Transformation* world2color, char cmd)
{
	const float T = 0.001f;
	const float R = 3.1415926f * 0.002f;
	Transformation trans;

	if (cmd == 'z') {
		trans.translation.y -= T;
	}
	if (cmd == 'x') {
		trans.translation.y += T;
	}
	if (cmd == 'c') {
		trans.translation.x -= T;
	}
	if (cmd == 'v') {
		trans.translation.x += T;
	}
	if (cmd == 'b') {
		trans.translation.z -= T;
	}
	if (cmd == 'n') {
		trans.translation.z += T;
	}
	if (cmd == '5') {
		trans.rotation0 = make_float3(cos(-R), 0, -sin(-R));
		trans.rotation1 = make_float3(0, 1, 0);
		trans.rotation2 = make_float3(sin(-R), 0, cos(-R));
	}
	if (cmd == '6') {
		trans.rotation0 = make_float3(cos(R), 0, -sin(R));
		trans.rotation1 = make_float3(0, 1, 0);
		trans.rotation2 = make_float3(sin(R), 0, cos(R));
	}
	if (cmd == '7') {
		trans.rotation0 = make_float3(1, 0, 0);
		trans.rotation1 = make_float3(0, cos(-R), sin(-R));
		trans.rotation2 = make_float3(0, -sin(-R), cos(-R));
	}
	if (cmd == '8') {
		trans.rotation0 = make_float3(1, 0, 0);
		trans.rotation1 = make_float3(0, cos(R), sin(R));
		trans.rotation2 = make_float3(0, -sin(R), cos(R));
	}
	if (cmd == '9') {
		trans.rotation0 = make_float3(cos(-R), sin(-R), 0);
		trans.rotation1 = make_float3(-sin(-R), cos(-R), 0);
		trans.rotation2 = make_float3(0, 0, 1);
	}
	if (cmd == '0') {
		trans.rotation0 = make_float3(cos(R), sin(R), 0);
		trans.rotation1 = make_float3(-sin(R), cos(R), 0);
		trans.rotation2 = make_float3(0, 0, 1);
	}
	for (int i = 0; i < cameras; i++) {
		world2color[i] = world2color[i] * trans;
	}
}
