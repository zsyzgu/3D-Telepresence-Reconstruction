#include "Timer.h"
#include "PointCloudProcess.h"
#include "Kinect2Grabber.h"
#include "SceneRegistration.h"
#include <pcl/gpu/features/features.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/compression/octree_pointcloud_compression.h>

//#define CREATE_EXE

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
pcl::Kinect2Grabber* grabber;
Eigen::Matrix4f transformation;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sceneLocal;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sceneRemote;

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event) {
	if (event.getKeySym() == "r" && event.keyDown()) {
		SceneRegistration registration;
		transformation = registration.align(sceneRemote, sceneLocal);
	}
}

void start() {
	omp_set_num_threads(4);

	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	grabber = new pcl::Kinect2Grabber();
	sceneLocal = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	sceneRemote = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

	transformation.setIdentity();
	pcl::io::loadPCDFile("view_remote.pcd", *sceneRemote);
	pcl::io::loadPCDFile("view_local.pcd", *sceneLocal);

	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer->registerKeyboardCallback(keyboardEventOccurred);
}

void update() {
	viewer->spinOnce();

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	cloud = grabber->getPointCloud();
	PointCloudProcess::pointCloud2PCNormal(sceneLocal, cloud);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformedRemote(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::transformPointCloud(*sceneRemote, *transformedRemote, transformation);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mergeScene(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	PointCloudProcess::merge2PointClouds(mergeScene, sceneLocal, transformedRemote);

	pcl::copyPointCloud(*mergeScene, *cloud);	

	if (!viewer->updatePointCloud(cloud, "result")) {
		viewer->addPointCloud(cloud, "result");
	}
}

#ifdef CREATE_EXE
int main(int argc, char *argv[]) {
	start();

	while (!viewer->wasStopped()) {
		Timer timer;

		update();

		timer.outputTime();
	}

	return 0;
}
#else

const int BUFFER_SIZE = 12000000;

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
		Timer timer;

		viewer->spinOnce();

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

		cloud = grabber->getPointCloud();
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformedRemote(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
		PointCloudProcess::pointCloud2PCNormal(sceneLocal, cloud);
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mergeScene(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
		pcl::transformPointCloud(*sceneRemote, *transformedRemote, transformation);
		PointCloudProcess::merge2PointClouds(mergeScene, sceneLocal, transformedRemote);

		pcl::copyPointCloud(*mergeScene, *cloud);
		if (!viewer->updatePointCloud(cloud, "result")) {
			viewer->addPointCloud(cloud, "result");
		}

		int size = mergeScene->size();
		loadBuffer(buffer, &size, 4);
#pragma omp parallel for
		for (int i = 0; i < size; i++) {
			int id = i * 27 + 4;
			loadBuffer(buffer + id + 0, &(mergeScene->points[i].x), 4);
			loadBuffer(buffer + id + 4, &(mergeScene->points[i].y), 4);
			loadBuffer(buffer + id + 8, &(mergeScene->points[i].z), 4);
			loadBuffer(buffer + id + 12, &(mergeScene->points[i].r), 1);
			loadBuffer(buffer + id + 13, &(mergeScene->points[i].g), 1);
			loadBuffer(buffer + id + 14, &(mergeScene->points[i].b), 1);
			loadBuffer(buffer + id + 15, &(mergeScene->points[i].normal_x), 4);
			loadBuffer(buffer + id + 19, &(mergeScene->points[i].normal_y), 4);
			loadBuffer(buffer + id + 23, &(mergeScene->points[i].normal_z), 4);
		}

		timer.outputTime();

		return buffer;
	}
}
#endif
