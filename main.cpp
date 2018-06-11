#include "Timer.h"
#include "PointCloudProcess.h"
#include "SceneRegistration.h"
#include "TsdfVolume.h"
#include "Transmission.h"
#include "RealsenseGrabber.h"
#include <pcl/gpu/features/features.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/compression/octree_pointcloud_compression.h>
#include <pcl/gpu/utils/safe_call.hpp>
#include <pcl/console/parse.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/conversions.h>
#include <pcl/surface/vtk_smoothing/vtk_utils.h>
#include <windows.h>

#define CREATE_EXE
//#define TRANSMISSION

const int BUFFER_SIZE = 100000000;
byte* buffer = NULL;
RealsenseGrabber* grabber = NULL;
TsdfVolume* volume = NULL;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

UINT16** depthImages;
RGBQUAD** colorImages;
Transformation depthTrans[MAX_CAMERAS];

#ifdef TRANSMISSION
Transmission* transmission = NULL;
#endif

void registration() {
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr local = grabber->getPointCloud(depthList[0], colorList[0]);
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr remote = grabber->getPointCloud(depthList[1], colorList[1]);
	//transformationList[1] = SceneRegistration::align(local, remote);
}

void saveScene() {
	pcl::io::savePCDFileASCII("scene.pcd", *cloud);
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event) {
	if (event.getKeySym() == "r" && event.keyDown()) {
		registration();
	}
	if (event.getKeySym() == "s" && event.keyDown()) {
		saveScene();
	}
}

void startViewer() {
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer->registerKeyboardCallback(keyboardEventOccurred);
}

#ifdef TRANSMISSION
DWORD WINAPI TransmissionRecvThread(LPVOID pM)
{
#pragma omp parallel sections
{
	#pragma omp section
	{
		while (true) {
			Sleep(1);
			transmission->recvRGBD(depthList[1], colorList[1]);
		}
	}
}
	return 0;
}
#endif

void start() {
	omp_set_num_threads(4);
	omp_set_nested(6);
	cudaSetDevice(1);
	
	grabber = new RealsenseGrabber();
	cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
	for (int i = 0; i < MAX_CAMERAS; i++) {
		depthTrans[i].setIdentity();
	}
	volume = new TsdfVolume(2, 2, 2, 0, 0, 1);
	buffer = new byte[BUFFER_SIZE];

#ifdef TRANSMISSION
	transmission = new Transmission(true);
	CreateThread(NULL, 0, TransmissionRecvThread, NULL, 0, NULL);
#endif
}

void update() {
	Transformation* colorTrans;
	Intrinsics* depthIntrinsics;
	Intrinsics* colorIntrinsics;

	int cameras = grabber->getRGBD(depthImages, colorImages, colorTrans, depthIntrinsics, colorIntrinsics);

#ifdef TRANSMISSION
	transmission->sendRGBD(depthList[0], colorList[0]);
	volume->integrate(2, depthList, colorList, transformationList);
#else
	volume->integrate(cameras, depthImages, colorImages, depthTrans, colorTrans, depthIntrinsics, colorIntrinsics);
#endif

	volume->calnMesh(buffer);
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
#ifdef TRANSMISSION
	if (transmission != NULL) {
		delete transmission;
	}
#endif
}

#ifdef CREATE_EXE

int main(int argc, char *argv[]) {
	start();
	startViewer();

	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		Timer timer;

		update();
		cloud = volume->getPointCloudFromMesh(buffer);

		timer.outputTime();

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

	__declspec(dllexport) void callSaveScene() {
		saveScene();
	}

	__declspec(dllexport) void callStop() {
		stop();
	}
}
#endif
