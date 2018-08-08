#include "Timer.h"
#include "SceneRegistration.h"
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
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
Transformation* world2color = NULL;
Transformation* world2depth = NULL;
Transmission* transmission = NULL;
int cameras = 0;
float* depthImages_device;
RGBQUAD* colorImages_device;
Intrinsics* depthIntrinsics;
Intrinsics* colorIntrinsics;

void registration(int targetId = 0) {
	if (targetId == 0) {
		SceneRegistration::align(cameras, grabber, world2color);
	} else {
		SceneRegistration::align(cameras, grabber, world2color, targetId);
	}
	Configuration::saveExtrinsics(world2color);
}

void setOrigin() {
	SceneRegistration::setOrigin(cameras, grabber, world2color);
	Configuration::saveExtrinsics(world2color);
}

void saveBackground() {
#if CALIBRATION == false:
	grabber->saveBackground();
#endif
}

void adjustTransformation() {
	SceneRegistration::adjust(cameras, grabber, world2color);
	Configuration::saveExtrinsics(world2color);
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event) {
	if (event.getKeySym() == "r" && event.keyDown()) {
		registration();
	}
	if (event.getKeySym() == "o" && event.keyDown()) {
		setOrigin();
	}
	if (event.getKeySym() == "a" && event.keyDown()) {
		adjustTransformation();
	}
	if (event.getKeySym() == "b" && event.keyDown()) {
		saveBackground();
	}
	if (event.getKeySym() == "1" && event.keyUp()) {
		registration(1);
	}
	if (event.getKeySym() == "2" && event.keyUp()) {
		registration(2);
	}
	if (event.getKeySym() == "3" && event.keyUp()) {
		registration(3);
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
	world2color = new Transformation[MAX_CAMERAS];
	world2depth = new Transformation[MAX_CAMERAS];
	Configuration::loadExtrinsics(world2color);

#if CALIBRATION == false
	grabber->loadBackground();
#endif

#ifdef TRANSMISSION
	int delayFrame = Configuration::loadDelayFrame();
	transmission = new Transmission(IS_SERVER, delayFrame);
	grabber->setTransmission(transmission);
#endif

	cameras = grabber->getRGBD(depthImages_device, colorImages_device, world2depth, world2color, depthIntrinsics, colorIntrinsics);
}

void update() {
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			int remoteCameras = 0;
			if (transmission != NULL && transmission->isConnected) {
				transmission->sendFrame();
				remoteCameras = transmission->getFrame(depthImages_device + cameras * DEPTH_H * DEPTH_W, colorImages_device + cameras * COLOR_H * COLOR_W, world2depth + cameras, depthIntrinsics + cameras, colorIntrinsics + cameras);
			}

			volume->integrate(buffer, cameras + remoteCameras, cameras, depthImages_device, colorImages_device, world2depth, depthIntrinsics, colorIntrinsics);
			cameras = grabber->getRGBD(depthImages_device, colorImages_device, world2depth, world2color, depthIntrinsics, colorIntrinsics);
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
	if (world2color != NULL) {
		delete[] world2color;
	}
	if (world2depth != NULL) {
		delete[] world2depth;
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
