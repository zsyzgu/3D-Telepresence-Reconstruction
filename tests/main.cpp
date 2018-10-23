#include "Timer.h"
#include "Calibration.h"
#include "TsdfVolume.h"
#include "Transmission.h"
#include "RealsenseGrabber.h"
#include "Parameters.h"
#include "Configuration.h"
#include <pcl/visualization/cloud_viewer.h>
#include <windows.h>

byte* buffer = NULL;
RealsenseGrabber* grabber = NULL;
TsdfVolume* volume = NULL;
Calibration* calibration = NULL;
Transmission* transmission = NULL;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

void registration(int targetId = 0) {
	if (targetId == 0) {
		calibration->align(grabber);
	} else {
		calibration->align(grabber, targetId);
	}
	Configuration::saveExtrinsics(calibration->getExtrinsics());
}

void setOrigin() {
	calibration->setOrigin(grabber);
	Configuration::saveExtrinsics(calibration->getExtrinsics());
}

void saveBackground() {
	grabber->saveBackground();
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event) {
	char cmd = event.getKeySym()[0];
	if (cmd == 'r' && event.keyDown()) {
		registration();
	}
	if (cmd == 'o' && event.keyDown()) {
		setOrigin();
	}
	if (cmd == 'b' && event.keyDown()) {
		saveBackground();
	}
	if (cmd == '1' && event.keyUp()) {
		registration(1);
	}
	if (cmd == '2' && event.keyUp()) {
		registration(2);
	}
}

void startViewer() {	
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer->registerKeyboardCallback(keyboardEventOccurred);
}

void start() {
	cudaSetDevice(0);
	omp_set_num_threads(2);

	grabber = new RealsenseGrabber();
	cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
	volume = new TsdfVolume(2, 2, 2, 0, 0, 0);
	buffer = new byte[MAX_VERTEX * sizeof(Vertex)];
	calibration = new Calibration();
	Configuration::loadExtrinsics(calibration->getExtrinsics());

#if CALIBRATION == false
	grabber->loadBackground();
#endif

#ifdef TRANSMISSION
	int delayFrame = Configuration::loadDelayFrame();
	transmission = new Transmission(IS_SERVER, delayFrame);
	grabber->setTransmission(transmission);
#endif

	grabber->updateRGBD(calibration->getExtrinsics());
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

			volume->integrate(buffer, grabber, remoteCameras, calibration->getExtrinsics());
			grabber->updateRGBD(calibration->getExtrinsics());
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
	if (buffer != NULL) {
		delete[] buffer;
	}
	if (calibration != NULL) {
		delete[] calibration;
	}
	if (transmission != NULL) {
		delete transmission;
	}
	std::cout << "stopped" << std::endl;
}

#ifdef CREATE_EXE

int main(int argc, char *argv[]) {
	start();

	Timer timer;
	startViewer();
	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		timer.reset();
		update();
#if CALIBRATION == false
		timer.outputTime(10);
#endif

		cloud = volume->getPointCloudFromMesh(buffer);
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
		return buffer;
	}

	__declspec(dllexport) void callSaveBackground() {
		saveBackground();
	}

	__declspec(dllexport) void callStop() {
		stop();
	}
}
#endif
