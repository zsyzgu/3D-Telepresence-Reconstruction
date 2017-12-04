#include "Timer.h"
#include "Recognition.h"
#include "Kinect2Pcd.h"

typedef pcl::PointXYZRGB PointType;

void recognizeModelFromScene(char* modelFileName, char* sceneFileName);
void captureModelAndSceneByKinect(char* modelFileName, char* sceneFileName);

int main(int argc, char *argv[]) {
	//recognizeModelFromScene("kinect_model.pcd", "kinect_scene.pcd");
	captureModelAndSceneByKinect("kinect_model.pcd", "kinect_scene.pcd");

	return 0;
}

void recognizeModelFromScene(char* modelFileName, char* sceneFileName) {
	Recognition recognition;
	pcl::PointCloud<PointType>::Ptr model(new pcl::PointCloud<PointType>());
	pcl::PointCloud<PointType>::Ptr scene(new pcl::PointCloud<PointType>());
	if (pcl::io::loadPCDFile(modelFileName, *model) < 0) {
		std::cout << "Error loading model cloud" << std::endl;
		return;
	}
	if (pcl::io::loadPCDFile(sceneFileName, *scene) < 0) {
		std::cout << "Error loading scene cloud" << std::endl;
		return;
	}
	recognition.recognize(model, scene);

	pcl::PointCloud<PointType>::Ptr modelKeypoints = recognition.getModelKeypoints();
	pcl::PointCloud<PointType>::Ptr sceneKeypoints = recognition.getSceneKeypoints();
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations = recognition.getRototranslations();
	std::vector<pcl::Correspondences> clustered_corrs = recognition.getClusteredCorrs();

	pcl::visualization::PCLVisualizer viewer("Correspondence Grouping");
	viewer.addPointCloud(scene, "scene_cloud");

	pcl::PointCloud<PointType>::Ptr off_scene_model(new pcl::PointCloud<PointType>());
	pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints(new pcl::PointCloud<PointType>());

	//  We are translating the model so that it doesn't end in the middle of the scene representation
	pcl::transformPointCloud(*model, *off_scene_model, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));
	pcl::transformPointCloud(*modelKeypoints, *off_scene_model_keypoints, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));

	pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler(off_scene_model, 255, 255, 128);
	viewer.addPointCloud(off_scene_model, off_scene_model_color_handler, "off_scene_model");

	pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler(sceneKeypoints, 0, 0, 255);
	viewer.addPointCloud(sceneKeypoints, scene_keypoints_color_handler, "scene_keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

	pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler(off_scene_model_keypoints, 0, 0, 255);
	viewer.addPointCloud(off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");

	for (size_t i = 0; i < rototranslations.size(); ++i)
	{
		pcl::PointCloud<PointType>::Ptr rotated_model(new pcl::PointCloud<PointType>());
		pcl::transformPointCloud(*model, *rotated_model, rototranslations[i]);

		std::stringstream ss_cloud;
		ss_cloud << "instance" << i;

		pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler(rotated_model, 255, 0, 0);
		viewer.addPointCloud(rotated_model, rotated_model_color_handler, ss_cloud.str());

		for (size_t j = 0; j < clustered_corrs[i].size(); ++j)
		{
			std::stringstream ss_line;
			ss_line << "correspondence_line" << i << "_" << j;
			PointType& model_point = off_scene_model_keypoints->at(clustered_corrs[i][j].index_query);
			PointType& scene_point = sceneKeypoints->at(clustered_corrs[i][j].index_match);

			//  We are drawing a line for each pair of clustered correspondences found between the model and the scene
			viewer.addLine<PointType, PointType>(model_point, scene_point, 0, 255, 0, ss_line.str());
		}
	}

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}

void captureModelAndSceneByKinect(char* modelFileName, char* sceneFileName) {
	Kinect2Pcd kinect2Pcd;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);

	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		if (kinect2Pcd.isUpdated()) {
			pcl::PointCloud<PointType>::Ptr cloud = kinect2Pcd.getPointCloud();
			if (!viewer->updatePointCloud(cloud, "cloud")) {
				viewer->addPointCloud(cloud, "cloud");
			}
		}
	}
}
