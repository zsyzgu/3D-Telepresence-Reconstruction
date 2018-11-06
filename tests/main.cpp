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

int main(int argc, char *argv[]) {
	start();

	pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
	viewer.setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer.registerKeyboardCallback(keyboardEventOccurred);

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
