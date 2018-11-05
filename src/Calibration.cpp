#include "Calibration.h"
#include "Timer.h"
#include "Parameters.h"

Transformation Calibration::calnInv(Transformation T)
{
	T.output();
	cv::Mat rv(3, 3, CV_64FC1);
	cv::Mat tv(3, 1, CV_64FC1);
	double* rv_d = (double*)rv.data;
	double* tv_d = (double*)tv.data;
	rv_d[0] = T.rotation0.x, rv_d[1] = T.rotation0.y, rv_d[2] = T.rotation0.z;
	rv_d[3] = T.rotation1.x, rv_d[4] = T.rotation1.y, rv_d[5] = T.rotation1.z;
	rv_d[6] = T.rotation2.x, rv_d[7] = T.rotation2.y, rv_d[8] = T.rotation2.z;
	tv_d[0] = T.translation.x, tv_d[1] = T.translation.y, tv_d[2] = T.translation.z;
	rv = rv.inv();
	tv = -rv * tv;
	return Transformation((double*)rv.data, (double*)tv.data);
}

void Calibration::rgb2mat(cv::Mat* mat, RGBQUAD* rgb)
{
	for (int i = 0; i < COLOR_H; i++) {
		for (int j = 0; j < COLOR_W; j++) {
			RGBQUAD color = rgb[i * COLOR_W + j];
			mat->at<cv::Vec3b>(i, j) = cv::Vec3b(color.rgbRed, color.rgbGreen, color.rgbBlue);
		}
	}
}

void Calibration::updateWorld2Depth(int cameras, RealsenseGrabber* grabber) {
	Transformation* color2depth = grabber->getColor2Depth();
	for (int i = 0; i < cameras; i++) {
		world2depth[i] = color2depth[i] * world2color[i];
	}
	Configuration::saveExtrinsics(world2depth);
}

Calibration::Calibration() {
	world2color = new Transformation[MAX_CAMERAS];
	world2depth = new Transformation[MAX_CAMERAS];
	Configuration::loadExtrinsics(world2depth);
}

Calibration::~Calibration() {
	if (world2color != NULL) {
		delete[] world2color;
	}
	if (world2depth != NULL) {
		delete[] world2depth;
	}
}

void Calibration::setOrigin(RealsenseGrabber* grabber) {
	int cameras = grabber->getCameras();

	std::vector<cv::Point3f> objectPoints;
	for (int r = 0; r < BOARD_SIZE.height; r++) {
		for (int c = 0; c < BOARD_SIZE.width; c++) {
			objectPoints.push_back(cv::Point3f(c * GRID_SIZE, r * GRID_SIZE, 0));
		}
	}

	Intrinsics* colorIntrinsics = grabber->getOriginColorIntrinsics();
	RGBQUAD** colorImages;
	std::vector<cv::Point2f> sourcePoints;
	cv::Mat sourceColorMat(COLOR_H, COLOR_W, CV_8UC3);

	int mainId = 0;
	do {
		grabber->getRGB(colorImages);
		rgb2mat(&sourceColorMat, colorImages[mainId]);
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
	Transformation camera0Inv = calnInv(world2color[mainId]);

	for (int i = 0; i < cameras; i++) {
		world2color[i] = (world2color[i] * camera0Inv) * world2camera;
	}
	updateWorld2Depth(cameras, grabber);

	cv::destroyAllWindows();
}

void Calibration::align(RealsenseGrabber* grabber, int targetId)
{
	int cameras = grabber->getCameras();
	assert(0 < targetId && targetId < cameras);

	RGBQUAD** colorImages;
	Intrinsics* colorIntrinsics = grabber->getOriginColorIntrinsics();
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
		grabber->getRGB(colorImages);
		rgb2mat(&sourceColorMat, colorImages[0]);
		rgb2mat(&targetColorMat, colorImages[targetId]);

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
		cv::pyrDown(mergeImage, mergeImage, cv::Size(mergeImage.cols / 2, mergeImage.rows / 2));
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
	updateWorld2Depth(cameras, grabber);
	cv::destroyAllWindows();
}

void Calibration::align(RealsenseGrabber* grabber) {
	int cameras = grabber->getCameras();
	world2color[0].setIdentity();
	for (int targetId = 1; targetId < cameras; targetId++) {
		align(grabber, targetId);
	}
}
