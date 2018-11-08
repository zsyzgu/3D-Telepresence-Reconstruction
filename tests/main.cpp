#include "Timer.h"
#include "TeleCP.h"
#include <pcl/visualization/cloud_viewer.h>

RealsenseGrabber* grabber = NULL;
TsdfVolume* volume = NULL;
Calibration* calibration = NULL;
Transmission* transmission = NULL;

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event) {
	char cmd = event.getKeySym()[0];
	if (cmd == 'r' && event.keyDown()) {
		calibration->align(grabber);
	}
	if (cmd == 'o' && event.keyDown()) {
		calibration->setOrigin(grabber);
	}
	if (cmd == 'b' && event.keyDown()) {
		grabber->saveBackground();
	}
	if (cmd == '1' && event.keyUp()) {
		calibration->align(grabber, 1);
	}
	if (cmd == '2' && event.keyUp()) {
		calibration->align(grabber, 2);
	}
}

void start() {
	cudaSetDevice(0);
	omp_set_num_threads(2);

	grabber = new RealsenseGrabber();
	volume = new TsdfVolume(2, 2, 2, 0, 0, 0);
	calibration = new Calibration();

	grabber->updateRGBD();

	if (transmission != NULL && transmission->isConnected) {
		transmission->prepareSendFrame(grabber, calibration->getExtrinsics());
	}
}

void update() {
#pragma omp parallel sections
	{
#pragma omp section
		{
			int remoteCameras = 0;
			if (transmission != NULL && transmission->isConnected) {
				transmission->sendFrame();
				remoteCameras = transmission->getFrame(grabber, calibration->getExtrinsics() + grabber->getCameras());
			}

			volume->integrate(grabber, remoteCameras, calibration->getExtrinsics());
			grabber->updateRGBD();

			if (transmission != NULL && transmission->isConnected) {
				transmission->prepareSendFrame(grabber, calibration->getExtrinsics());
			}
		}
#pragma omp section
		{
			if (transmission != NULL && transmission->isConnected) {
				transmission->recvFrame();
			}
		}
	}
}

void stop() {
	if (grabber != NULL) {
		delete grabber;
	}
	if (volume != NULL) {
		delete volume;
	}
	if (calibration != NULL) {
		delete[] calibration;
	}
	if (transmission != NULL) {
		delete transmission;
	}
	std::cout << "stopped" << std::endl;
}

#include <pcl/registration/gicp6d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
Eigen::Matrix4f extrinsics2Mat4(Extrinsics extrinsics) {
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
Extrinsics calnInv(Extrinsics T)
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
int main(int argc, char *argv[]) {
	pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
	viewer.setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer.registerKeyboardCallback(keyboardEventOccurred);

	/*pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGBA>());
	pcl::io::loadPCDFile("source.pcd", *source);
	pcl::io::loadPCDFile("target.pcd", *target);

	Extrinsics extrinsics[MAX_CAMERAS];
	Configuration::loadExtrinsics(extrinsics);
	Eigen::Matrix4f world2depthSource = extrinsics2Mat4(calnInv(extrinsics[0]));
	Eigen::Matrix4f world2depthTarget = extrinsics2Mat4(calnInv(extrinsics[1]));

	pcl::transformPointCloud(*source, *source, world2depthSource);
	pcl::transformPointCloud(*target, *target, world2depthTarget);

	pcl::GeneralizedIterativeClosestPoint6D gicp;
	gicp.setInputSource(source);
	gicp.setInputTarget(target);
	gicp.setMaximumIterations(50);
	gicp.setTransformationEpsilon(1e-8);
	gicp.align(*output);
	std::cout << gicp.getFitnessScore() << std::endl;
	Eigen::Matrix4f adjustment = gicp.getFinalTransformation();

	pcl::transformPointCloud(*source, *source, adjustment);

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> sourceRGB(source, 0, 0, 255);
	viewer.addPointCloud<pcl::PointXYZRGBA>(source, sourceRGB, "source");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> targetRGB(target, 0, 255, 0);
	viewer.addPointCloud<pcl::PointXYZRGBA>(target, targetRGB, "target");

	while (!viewer.wasStopped()) {
		viewer.spinOnce();
	}*/

	start();

	while (!viewer.wasStopped()) {
		viewer.spinOnce();

		Timer timer;
		update();
		timer.outputTime();

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = volume->getPointCloud();
		if (!viewer.updatePointCloud(cloud, "cloud")) {
			viewer.addPointCloud(cloud, "cloud");
		}
	}

	stop();

	return 0;
}
