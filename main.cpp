#include "Timer.h"
#include "PointCloudProcess.h"
#include "Kinect2Grabber.h"
#include "SceneRegistration.h"
#include "TsdfVolume.h"
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

//#define CREATE_EXE

const int BUFFER_SIZE = 30000000;
byte* buffer;
pcl::Kinect2Grabber* grabber;
TsdfVolume* volume;
Eigen::Matrix4f transformation;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

void registration() {

}

void setBackground() {
	grabber->updateBackground();
}

void saveScene() {
	pcl::io::savePCDFileASCII("scene.pcd", *cloud);
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event) {
	if (event.getKeySym() == "r" && event.keyDown()) {
		registration();
	}
	if (event.getKeySym() == "b" && event.keyDown()) {
		setBackground();
	}
	if (event.getKeySym() == "s" && event.keyDown()) {
		saveScene();
	}
}

void start() {
	cudaSetDevice(1);
	omp_set_num_threads(4);
	omp_set_nested(6);

	grabber = new pcl::Kinect2Grabber();
	cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());

	transformation.setIdentity();

	volume = new TsdfVolume(512, 512, 512, 1, 1, 1, 0, 0, 0.5);

	buffer = new byte[BUFFER_SIZE];
}

void update() {
	grabber->updateDepthAndColor();
	UINT16* depthData = grabber->getDepthData();
	RGBQUAD* colorData = grabber->getColorData();
	volume->integrate(depthData, colorData, transformation);
	volume->calnMesh(buffer);
}

void stop() {
	if (grabber != NULL) {
		delete grabber;
	}
	if (volume != NULL) {
		delete volume;
	}
	if (buffer != nullptr) {
		delete[] buffer;
	}
}

#ifdef CREATE_EXE
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

void startViewer() {
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer->registerKeyboardCallback(keyboardEventOccurred);
}

int main(int argc, char *argv[]) {
	start();
	startViewer();

	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		Timer timer;

		update();

		timer.outputTime();

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

	__declspec(dllexport) void callSetBackground() {
		setBackground();
	}

	__declspec(dllexport) void callSaveScene() {
		saveScene();
	}

	__declspec(dllexport) void callStop() {
		stop();
	}
}
#endif
