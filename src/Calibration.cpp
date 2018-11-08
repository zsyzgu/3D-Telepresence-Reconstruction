#include "Calibration.h"
#include "Timer.h"
#include "Parameters.h"
#include <pcl/registration/gicp6d.h>
#include <pcl/io/pcd_io.h>

void Calibration::initCheckerboardPoints()
{
	checkerboardPoints.clear();
	for (int r = 0; r < BOARD_SIZE.height; r++) {
		for (int c = 0; c < BOARD_SIZE.width; c++) {
			checkerboardPoints.push_back(cv::Point3f(c * GRID_SIZE, r * GRID_SIZE, 0));
		}
	}
}

Extrinsics Calibration::calnInv(Extrinsics T)
{
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
	return Extrinsics((double*)rv.data, (double*)tv.data);
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

cv::Mat Calibration::intrinsics2mat(Intrinsics I)
{
	cv::Mat mat(cv::Size(3, 3), CV_32F);
	mat.at<float>(0, 0) = I.fx;
	mat.at<float>(1, 1) = I.fy;
	mat.at<float>(0, 2) = I.ppx;
	mat.at<float>(1, 2) = I.ppy;
	mat.at<float>(2, 2) = 1;
	return mat;
}

Eigen::Matrix4f Calibration::extrinsics2Mat4(Extrinsics extrinsics) {
	Eigen::Matrix4f mat;
	float* data = mat.data();
	data[0] = extrinsics.rotation0.x;
	data[1] = extrinsics.rotation1.x;
	data[2] = extrinsics.rotation2.x;
	data[4] = extrinsics.rotation0.y;
	data[5] = extrinsics.rotation1.y;
	data[6] = extrinsics.rotation2.y;
	data[8] = extrinsics.rotation0.z;
	data[9] = extrinsics.rotation1.z;
	data[10] = extrinsics.rotation2.z;
	data[12] = extrinsics.translation.x;
	data[13] = extrinsics.translation.y;
	data[14] = extrinsics.translation.z;
	return mat;
}

void Calibration::updateWorld2Depth(int cameraId, RealsenseGrabber* grabber) {
	assert(0 <= cameraId && cameraId < MAX_CAMERAS);
	Extrinsics* color2depth = grabber->getColor2Depth();
	world2depth[cameraId] = color2depth[cameraId] * world2color[cameraId];
	Configuration::saveExtrinsics(world2depth);
}

void Calibration::icpWorld2Depth(int cameraId, RealsenseGrabber* grabber)
{
	assert(0 < cameraId && cameraId < MAX_CAMERAS);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGBA>());
	float* depthSource = grabber->getDepthImages_host();
	float* depthTarget = depthSource + cameraId * DEPTH_H * DEPTH_W;
	RGBQUAD* colorSource = grabber->getColorImages_host();
	RGBQUAD* colorTarget = colorSource + cameraId * COLOR_H * COLOR_W;
	Intrinsics* intrinsicsDepthSource = grabber->getDepthIntrinsics();
	Intrinsics* intrinsicsDepthTarget = intrinsicsDepthSource + cameraId;
	Intrinsics* intrinsicsColorSource = grabber->getColorIntrinsics();
	Intrinsics* intrinsicsColorTarget = intrinsicsColorSource + cameraId;

	for (int y = 0; y < DEPTH_H; y++) {
		for (int x = 0; x < DEPTH_W; x++) {
			int id = y * DEPTH_W + x;
			if (depthSource[id] != 0 && depthSource[id] < BOARD_MAX_DISTANCE) {
				float3 pos = intrinsicsDepthSource->deproject(make_float2(x, y), depthSource[id]);
				int2 pos2d = intrinsicsColorSource->translate(pos);
				if (0 <= pos2d.x && pos2d.x < COLOR_W && 0 <= pos2d.y && pos2d.y < COLOR_H) {
					RGBQUAD rgb = colorSource[pos2d.y * COLOR_W + pos2d.x];
					pcl::PointXYZRGBA point;
					point.x = pos.x;
					point.y = pos.y;
					point.z = pos.z;
					point.r = rgb.rgbRed;
					point.g = rgb.rgbGreen;
					point.b = rgb.rgbBlue;
					source->push_back(point);
				}
			}
			if (depthTarget[id] != 0 && depthTarget[id] < BOARD_MAX_DISTANCE) {
				float3 pos = intrinsicsDepthTarget->deproject(make_float2(x, y), depthTarget[id]);
				int2 pos2d = intrinsicsColorTarget->translate(pos);
				if (0 <= pos2d.x && pos2d.x < COLOR_W && 0 <= pos2d.y && pos2d.y < COLOR_H) {
					RGBQUAD rgb = colorTarget[pos2d.y * COLOR_W + pos2d.x];
					pcl::PointXYZRGBA point;
					point.x = pos.x;
					point.y = pos.y;
					point.z = pos.z;
					point.r = rgb.rgbRed;
					point.g = rgb.rgbGreen;
					point.b = rgb.rgbBlue;
					target->push_back(point);
				}
			}
		}
	}

	Eigen::Matrix4f depth2worldSource = extrinsics2Mat4(calnInv(world2depth[0]));
	Eigen::Matrix4f depth2worldTarget = extrinsics2Mat4(calnInv(world2depth[cameraId]));

	pcl::transformPointCloud(*source, *source, depth2worldSource);
	pcl::transformPointCloud(*target, *target, depth2worldTarget);

	std::cout << "GICP..." << std::endl;
	pcl::GeneralizedIterativeClosestPoint6D gicp;
	gicp.setInputSource(source); // align cameras[cameraId] to cameras[0]
	gicp.setInputTarget(target);
	gicp.setMaximumIterations(50);
	gicp.setTransformationEpsilon(1e-8);
	gicp.align(*output);
	std::cout << "Fitness Score = " << gicp.getFitnessScore() << std::endl;
	Eigen::Matrix4f adjustment = gicp.getFinalTransformation();
	Extrinsics source2target = Extrinsics((float*)adjustment.data());
	source2target.output();
	world2depth[cameraId] = world2depth[cameraId] * source2target;

	Configuration::saveExtrinsics(world2depth);
}

Calibration::Calibration() {
	world2color = new Extrinsics[MAX_CAMERAS];
	world2depth = new Extrinsics[MAX_CAMERAS];
	Configuration::loadExtrinsics(world2depth);
	initCheckerboardPoints();
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

	Intrinsics* colorIntrinsics = grabber->getOriginColorIntrinsics();
	std::vector<cv::Point2f> sourcePoints;
	cv::Mat sourceColorMat(COLOR_H, COLOR_W, CV_8UC3);

	int mainId = 0;
	do {
		grabber->updateRGBD();
		RGBQUAD* originColorImages = grabber->getOriginColorImages_host();

		rgb2mat(&sourceColorMat, originColorImages + mainId * COLOR_H * COLOR_W);
		sourcePoints.clear();
		findChessboardCorners(sourceColorMat, BOARD_SIZE, sourcePoints, /*cv::CALIB_CB_ADAPTIVE_THRESH | */cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

		cv::Scalar color = cv::Scalar(0, 0, 255);
		if (sourcePoints.size() == BOARD_NUM) {
			color = cv::Scalar(0, 255, 255);
		}
		for (int i = 0; i < sourcePoints.size(); i++) {
			cv::circle(sourceColorMat, sourcePoints[i], 3, (i == 0) ? cv::Scalar(255, 0, 0) : color, 2);
		}
		cv::imshow("Get Depth", sourceColorMat);
		char ch = cv::waitKey(1);
		if ('0' <= ch && ch < '4') {
			mainId = ch - '0';
		}

	} while (sourcePoints.size() != BOARD_NUM);

	cv::Mat sourceCameraMatrix = intrinsics2mat(colorIntrinsics[mainId]);
	cv::Mat distCoeffs;
	cv::Mat rv(3, 1, CV_64FC1);
	cv::Mat tv(3, 1, CV_64FC1);
	solvePnP(checkerboardPoints, sourcePoints, sourceCameraMatrix, distCoeffs, rv, tv);
	cv::Rodrigues(rv, rv);
	Extrinsics world2camera((double*)rv.data, (double*)tv.data);
	Extrinsics camera0Inv = calnInv(world2color[mainId]);

	for (int i = 0; i < cameras; i++) {
		world2color[i] = (world2color[i] * camera0Inv) * world2camera;
		updateWorld2Depth(i, grabber);
	}

	cv::destroyAllWindows();
}

void Calibration::align(RealsenseGrabber* grabber, int targetId)
{
	int cameras = grabber->getCameras();
	assert(0 < targetId && targetId < cameras);

	Intrinsics* colorIntrinsics = grabber->getOriginColorIntrinsics();
	std::vector<cv::Point2f> sourcePoints;
	std::vector<cv::Point2f> targetPoints;
	cv::Mat sourceColorMat(COLOR_H, COLOR_W, CV_8UC3);
	cv::Mat targetColorMat(COLOR_H, COLOR_W, CV_8UC3);

	std::vector<std::vector<cv::Point2f> > sourcePointsArray;
	std::vector<std::vector<cv::Point2f> > targetPointsArray;
	std::vector<cv::Point2f> rects;
	for (int iter = 0; iter < ITERATION;) {
		grabber->updateRGBD();
		RGBQUAD* originColorImages = grabber->getOriginColorImages_host();

		rgb2mat(&sourceColorMat, originColorImages);
		rgb2mat(&targetColorMat, originColorImages + targetId * COLOR_H * COLOR_W);

		sourcePoints.clear();
		targetPoints.clear();
		findChessboardCorners(sourceColorMat, BOARD_SIZE, sourcePoints, /*cv::CALIB_CB_ADAPTIVE_THRESH | */cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
		findChessboardCorners(targetColorMat, BOARD_SIZE, targetPoints, /*cv::CALIB_CB_ADAPTIVE_THRESH | */cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
		cv::Scalar color = cv::Scalar(0, 0, 255);
		if (sourcePoints.size() == BOARD_NUM) {
			color = cv::Scalar(0, 255, 0);
		}
		for (int i = 0; i < sourcePoints.size(); i++) {
			cv::circle(sourceColorMat, sourcePoints[i], 3, (i == 0) ? cv::Scalar(255, 0, 0) : color, 2);
		}
		for (int i = 0; i < targetPoints.size(); i++) {
			cv::circle(targetColorMat, targetPoints[i], 3, (i == 0) ? cv::Scalar(255, 0, 0) : color, 2);
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

	std::vector<std::vector<cv::Point3f> > checkerboardPointsArray(sourcePointsArray.size(), checkerboardPoints);
	cv::Mat sourceCameraMatrix = intrinsics2mat(colorIntrinsics[0]);
	cv::Mat targetCameraMatrix = intrinsics2mat(colorIntrinsics[targetId]);
	cv::Mat sourceDistCoeffs;
	cv::Mat targetDistCoeffs;
	cv::Mat rotation, translation, essential, fundamental;
	double rms = cv::stereoCalibrate(
		checkerboardPointsArray,
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

	world2color[targetId] = Extrinsics((double*)rotation.data, (double*)translation.data);
	updateWorld2Depth(targetId, grabber); // caln world2depth from world2color and color2depth
	icpWorld2Depth(targetId, grabber); // using ICP and points cloud to adjust the calibration
	cv::destroyAllWindows();
}

void Calibration::align(RealsenseGrabber* grabber) {
	int cameras = grabber->getCameras();
	world2color[0].setIdentity();
	updateWorld2Depth(0, grabber);
	for (int targetId = 1; targetId < cameras; targetId++) {
		align(grabber, targetId);
	}
}
