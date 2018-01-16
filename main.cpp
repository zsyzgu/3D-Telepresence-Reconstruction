#include "Timer.h"
#include "Recognition.h"
#include "PointCloudProcess.h"
#include "Kinect2Grabber.h"
#include "pcl/gpu/features/features.hpp"

// ===== Recognize Model from Scene =====
void recognizeModelFromScene(char* modelFileName, char* sceneFileName);

// ===== Capture Model and Scene by Kinect =====
void captureModelAndSceneByKinect(char* modelFileName, char* sceneFileName);
int capturePointCnt = 0;
Eigen::Vector2f captureWindowMin;
Eigen::Vector2f captureWindowMax;
void mouseEventOccurred(const pcl::visualization::MouseEvent &event, void* viewerVoid);

int main(int argc,				 char *argv[]) {
	//recognizeModelFromScene("chair.pcd", "scene.pcd");
	//captureModelAndSceneByKinect("model.pcd", "scene.pcd");
	//merge2PointClouds("model1.pcd", "model2.pcd");


	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sceneRemote(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::io::loadPCDFile("scene_remote.pcd", *sceneRemote);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::Kinect2Grabber grabber;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudNormals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		Timer timer;
		timer.reset();

		cloud = grabber.getPointCloud();

		PointCloudProcess::pointCloud2PCNormal(cloudNormals, cloud);
		PointCloudProcess::merge2PointClouds(result, cloudNormals, sceneRemote);
		
		timer.outputTime();

		pcl::copyPointCloud(*result, *cloud);
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

	pcl::io::savePCDFileASCII("scene.pcd", *result);

	return 0;
}

void recognizeModelFromScene(char* modelFileName, char* sceneFileName) {
	Recognition recognition;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr model(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>());
	if (pcl::io::loadPCDFile(modelFileName, *model) < 0) {
		std::cout << "Error loading model cloud" << std::endl;
		return;
	}
	if (pcl::io::loadPCDFile(sceneFileName, *scene) < 0) {
		std::cout << "Error loading scene cloud" << std::endl;
		return;
	}
	recognition.recognize(model, scene);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr modelKeypoints = recognition.getModelKeypoints();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sceneKeypoints = recognition.getSceneKeypoints();
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations = recognition.getRototranslations();
	std::vector<pcl::Correspondences> clustered_corrs = recognition.getClusteredCorrs();

	pcl::visualization::PCLVisualizer viewer("Correspondence Grouping");
	viewer.addPointCloud(scene, "scene_cloud");

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr off_scene_model(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr off_scene_model_keypoints(new pcl::PointCloud<pcl::PointXYZRGB>());

	//  We are translating the model so that it doesn't end in the middle of the scene representation
	pcl::transformPointCloud(*model, *off_scene_model, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));
	pcl::transformPointCloud(*modelKeypoints, *off_scene_model_keypoints, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> off_scene_model_color_handler(off_scene_model, 255, 255, 128);
	viewer.addPointCloud(off_scene_model, off_scene_model_color_handler, "off_scene_model");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> scene_keypoints_color_handler(sceneKeypoints, 0, 0, 255);
	viewer.addPointCloud(sceneKeypoints, scene_keypoints_color_handler, "scene_keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> off_scene_model_keypoints_color_handler(off_scene_model_keypoints, 0, 0, 255);
	viewer.addPointCloud(off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");

	for (size_t i = 0; i < rototranslations.size(); ++i)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr rotated_model(new pcl::PointCloud<pcl::PointXYZRGB>());
		pcl::transformPointCloud(*model, *rotated_model, rototranslations[i]);

		std::stringstream ss_cloud;
		ss_cloud << "instance" << i;

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> rotated_model_color_handler(rotated_model, 255, 0, 0);
		viewer.addPointCloud(rotated_model, rotated_model_color_handler, ss_cloud.str());

		for (size_t j = 0; j < clustered_corrs[i].size(); ++j)
		{
			std::stringstream ss_line;
			ss_line << "correspondence_line" << i << "_" << j;
			pcl::PointXYZRGB& model_point = off_scene_model_keypoints->at(clustered_corrs[i][j].index_query);
			pcl::PointXYZRGB& scene_point = sceneKeypoints->at(clustered_corrs[i][j].index_match);

			//  We are drawing a line for each pair of clustered correspondences found between the model and the scene
			viewer.addLine<pcl::PointXYZRGB, pcl::PointXYZRGB>(model_point, scene_point, 0, 255, 0, ss_line.str());
		}
	}

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
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

/*void merge2PointClouds(char* model1FileName, char* model2FileName)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZRGB>());
	if (pcl::io::loadPCDFile(model1FileName, *cloud1) < 0) {
		std::cout << "Error loading model 1" << std::endl;
	}
	PointCloudProcess::mlsFiltering(cloud1);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGB>());
	if (pcl::io::loadPCDFile(model2FileName, *cloud2) < 0) {
		std::cout << "Error loading model 2" << std::endl;
	}
	PointCloudProcess::mlsFiltering(cloud2);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	PointCloudProcess::merge2PointClouds(cloud, cloud1, cloud2);
	PointCloudProcess::mlsFiltering(cloud);

	pcl::visualization::PCLVisualizer viewer("Camera");
	viewer.addPointCloud(cloud, "model");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "model");

	while (viewer.wasStopped() == false) {
		viewer.spinOnce();
	}
}*/
