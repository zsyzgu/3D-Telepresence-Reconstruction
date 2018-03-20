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

#define CREATE_EXE

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
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
}

void startViewer() {
	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer->registerKeyboardCallback(keyboardEventOccurred);
}

void update() {
	grabber->updateDepthAndColor();

	UINT16* depthData = grabber->getDepthData();
	RGBQUAD* colorData = grabber->getColorData();

	volume->integrate(depthData, colorData, transformation);

	cloud = volume->calnMesh();
}

#ifdef CREATE_EXE
int main(int argc, char *argv[]) {
	start();
	startViewer();

	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		update();

		if (!viewer->updatePointCloud(cloud, "cloud")) {
			viewer->addPointCloud(cloud, "cloud");
		}
	}

	return 0;
}

#else
const int BUFFER_SIZE = 30000000;

byte buffer[BUFFER_SIZE];

void loadBuffer(byte* dst, void* src, int size) {
	byte* pt = (byte*)src;
	for (int i = 0; i < size; i++) {
		dst[i] = pt[i];
	}
}

extern "C" {
	__declspec(dllexport) void callStart() {
		start();
	}

	__declspec(dllexport) byte* callUpdate() {
		update();

		int size = cloud->size();
		loadBuffer(buffer, &size, 4);
#pragma omp parallel for
		for (int i = 0; i < size; i++) {
			int id = i * 15 + 4;
			loadBuffer(buffer + id + 0, &(cloud->points[i].x), 4);
			loadBuffer(buffer + id + 4, &(cloud->points[i].y), 4);
			loadBuffer(buffer + id + 8, &(cloud->points[i].z), 4);
			loadBuffer(buffer + id + 12, &(cloud->points[i].r), 1);
			loadBuffer(buffer + id + 13, &(cloud->points[i].g), 1);
			loadBuffer(buffer + id + 14, &(cloud->points[i].b), 1);
		}

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

	}
}
#endif
