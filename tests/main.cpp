#include "Timer.h"
#include "Calibration.h"
#include "TsdfVolume.h"
#include "Transmission.h"
#include "RealsenseGrabber.h"
#include "Parameters.h"
#include "Configuration.h"
#include <pcl/visualization/cloud_viewer.h>
#include <windows.h>

RealsenseGrabber* grabber = NULL;
TsdfVolume* volume = NULL;
Calibration* calibration = NULL;
Transmission* transmission = NULL;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

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

void startViewer() {	
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer->registerKeyboardCallback(keyboardEventOccurred);
	cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
}

void start() {
	cudaSetDevice(0);
	omp_set_num_threads(2);

	grabber = new RealsenseGrabber();
	volume = new TsdfVolume(2, 2, 2, 0, 0, 0);
	calibration = new Calibration();

#if CALIBRATION == false
	grabber->loadBackground();
#endif

#if TRANSMISSION == true
	transmission = new Transmission(IS_SERVER);
	grabber->setTransmission(transmission);
#endif

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

#if CREATE_EXE == true

int main(int argc, char *argv[]) {
	start();

	Timer timer;
	startViewer();
	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		timer.reset();
		update();
		timer.outputTime(10);

		cloud = volume->getPointCloud();
		if (!viewer->updatePointCloud(cloud, "cloud")) {
			viewer->addPointCloud(cloud, "cloud");
		}
	}

	stop();
	return 0;
}

#else
extern "C" {
	__declspec(dllexport) void callStart() {
		start();
	}

	__declspec(dllexport) byte* callUpdate() {
		update();
		return volume->getBuffer();
	}

	__declspec(dllexport) void callStop() {
		stop();
	}
}
#endif
