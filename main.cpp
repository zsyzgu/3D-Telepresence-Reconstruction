#include "Timer.h"
#include "PointCloudProcess.h"
#include "Kinect2Grabber.h"
#include "SceneRegistration.h"
#include <pcl/gpu/features/features.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>

// ===== Capture Model and Scene by Kinect =====
void captureModelAndSceneByKinect(char* modelFileName, char* sceneFileName);
int capturePointCnt = 0;
Eigen::Vector2f captureWindowMin;
Eigen::Vector2f captureWindowMax;
void mouseEventOccurred(const pcl::visualization::MouseEvent &event, void* viewerVoid);

Eigen::Matrix4f transformation;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sceneLocal(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sceneRemote(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event) {
	if (event.getKeySym() == "r" && event.keyDown()) {
		SceneRegistration registration;
		transformation = registration.align(sceneRemote, sceneLocal);
	}
}

int main(int argc, char *argv[]) {
	//recognizeModelFromScene("view1.pcd", "view2.pcd");
	//captureModelAndSceneByKinect("model.pcd", "scene.pcd");
	//merge2PointClouds("model1.pcd", "model2.pcd");
	omp_set_num_threads(4);

	transformation.setIdentity();
	pcl::io::loadPCDFile("view_remote.pcd", *sceneRemote);
	pcl::io::loadPCDFile("view_local.pcd", *sceneLocal);

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mergeScene(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformedRemote(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

	pcl::Kinect2Grabber grabber;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer->registerKeyboardCallback(keyboardEventOccurred);

	Timer timer;
	timer.reset();
	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

		cloud = grabber.getPointCloud();
		PointCloudProcess::pointCloud2PCNormal(sceneLocal, cloud);
		pcl::transformPointCloud(*sceneRemote, *transformedRemote, transformation);
		PointCloudProcess::merge2PointClouds(mergeScene, sceneLocal, transformedRemote);

		pcl::copyPointCloud(*mergeScene, *cloud);
		if (!viewer->updatePointCloud(cloud, "result")) {
			viewer->addPointCloud(cloud, "result");
		}

		timer.outputTime(10);
		timer.reset();
	}

	/*pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::Kinect2Grabber grabber;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);

	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		cloud = grabber.getPointCloud();

		if (!viewer->updatePointCloud(cloud, "cloud")) {
			viewer->addPointCloud(cloud, "cloud");
		}

#if false
		viewer->removePointCloud("normal", 0);
		if (cloudNormals->size() > 0) {
			viewer->addPointCloudNormals<pcl::PointXYZRGBNormal>(result, 20, 0.03, "normal");
		}
#endif
	}

	PointCloudProcess::pointCloud2PCNormal(result, cloud);

	pcl::io::savePCDFileASCII("scene.pcd", *result);*/

	return 0;
}

void captureModelAndSceneByKinect(char* modelFileName, char* sceneFileName) {
	const int screenWidth = 1280;
	const int screenHeight = 720;

	pcl::Kinect2Grabber grabber;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setSize(screenWidth, screenHeight);
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer->registerMouseCallback(mouseEventOccurred, (void*)viewer.get());

	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		scene = grabber.getPointCloud();
		if (!viewer->updatePointCloud(scene, "cloud")) {
			viewer->addPointCloud(scene, "cloud");
		}
	}

	pcl::visualization::Camera camera;
	viewer->getCameraParameters(camera);
	camera.window_size[0] = screenWidth;
	camera.window_size[1] = screenHeight;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr model(new pcl::PointCloud<pcl::PointXYZRGB>());

	for (int i = 0; i < scene->size() ; i++) {
		pcl::PointXYZRGB pt = scene->at(i);
		if (pt.x != 0) {
			Eigen::Vector4d windowCord;
			camera.cvtWindowCoordinates(pt, windowCord);
			float screenX = windowCord[0];
			float screenY = windowCord[1];
			if (captureWindowMin.x() <= screenX && screenX <= captureWindowMax.x() && captureWindowMin.y() <= screenY && screenY <= captureWindowMax.y()) {
				model->push_back(scene->at(i));
			}
		}
	}

	pcl::io::savePCDFileASCII(sceneFileName, *scene);
	pcl::io::savePCDFileASCII(modelFileName, *model);
}

void mouseEventOccurred(const pcl::visualization::MouseEvent &event, void* viewerVoid) {
	pcl::visualization::PCLVisualizer* viewer = static_cast<pcl::visualization::PCLVisualizer *>(viewerVoid);

	if (event.getButton() == pcl::visualization::MouseEvent::RightButton && event.getType() == pcl::visualization::MouseEvent::MouseButtonPress) {
		std::cout << "clicked: " << event.getX() << " " << event.getY() << std::endl;
		if ((capturePointCnt++) % 2 == 0) {
			captureWindowMin = Eigen::Vector2f(event.getX(), event.getY());
		} else {
			captureWindowMax = Eigen::Vector2f(event.getX(), event.getY());
		}
	}
}
