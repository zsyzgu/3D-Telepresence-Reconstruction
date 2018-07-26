#include "Timer.h"
#include "SceneRegistration.h"
#include "TsdfVolume.h"
#include "Transmission.h"
#include "RealsenseGrabber.h"
#include "Parameters.h"
#include "Configuration.h"
#include <pcl/visualization/cloud_viewer.h>
#include <windows.h>

#define CREATE_EXE
#define TRANSMISSION

byte* buffer = NULL;
RealsenseGrabber* grabber = NULL;
TsdfVolume* volume = NULL;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
Transformation* world2color = NULL;
Transmission* transmission = NULL;

void registration() {
	SceneRegistration::align(grabber, world2color);
}

void saveExtrinsics() {
	Configuration::saveExtrinsics(world2color);
}

void saveBackground() {
	grabber->saveBackground();
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event) {
	if (event.getKeySym() == "r" && event.keyDown()) {
		registration();
	}
	if (event.getKeySym() == "s" && event.keyDown()) {
		saveExtrinsics();
	}
	if (event.getKeySym() == "b" && event.keyDown()) {
		saveBackground();
	}
}

void startViewer() {
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer->registerKeyboardCallback(keyboardEventOccurred);
}

DWORD WINAPI TransmissionRecvThread(LPVOID pM)
{
	while (transmission != NULL) {
		Sleep(1);
	}
	return 0;
}

void start() {
	cudaSetDevice(0);
	
	grabber = new RealsenseGrabber();
	cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
	volume = new TsdfVolume(2, 2, 2, 0, 0, 1.);
	buffer = new byte[MAX_VERTEX * sizeof(Vertex)];
	world2color = new Transformation[MAX_CAMERAS];
	Configuration::loadExtrinsics(world2color);
	grabber->loadBackground();

#ifdef TRANSMISSION
	transmission = new Transmission(2);
	CreateThread(NULL, 0, TransmissionRecvThread, NULL, 0, NULL);
	grabber->setTransmission(transmission);
#endif
}

void update() {
	float* depthImages_device;
	RGBQUAD* colorImages_device;
	Transformation* color2depth;
	Intrinsics* depthIntrinsics;
	Intrinsics* colorIntrinsics;
	int cameras = grabber->getRGBD(depthImages_device, colorImages_device, color2depth, depthIntrinsics, colorIntrinsics);

	if (transmission != NULL) {
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				int remoteCameras = transmission->getFrame(depthImages_device + cameras * DEPTH_H * DEPTH_W, colorImages_device + cameras * COLOR_H * COLOR_W, color2depth + cameras, depthIntrinsics + cameras, colorIntrinsics + cameras);
				cameras += remoteCameras;
			}
			#pragma omp section
			{
				transmission->recvFrame();
			}
		}
		transmission->endFrame();
	}
	

	volume->integrate(buffer, cameras, depthImages_device, colorImages_device, color2depth, world2color, depthIntrinsics, colorIntrinsics);
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
	if (world2color != NULL) {
		delete[] world2color;
	}
	if (transmission != NULL) {
		delete transmission;
	}
}

#ifdef CREATE_EXE

int main(int argc, char *argv[]) {
	start();
	startViewer();

	Timer timer;

	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		timer.reset();
		update();
		timer.outputTime(10);

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

	__declspec(dllexport) void callRegistration() {
		registration();
	}

	__declspec(dllexport) void callSaveExtrinsics() {
		saveExtrinsics();
	}

	__declspec(dllexport) void callSaveBackground() {
		saveBackground();
	}

	__declspec(dllexport) void callStop() {
		stop();
	}
}
#endif
