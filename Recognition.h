#ifndef RECOGNITION_H
#define RECOGNITION_H

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>

class Recognition {
private:
	typedef pcl::PointXYZRGB PointType;
	typedef pcl::Normal NormalType;
	typedef pcl::ReferenceFrame RFType;
	typedef pcl::SHOT352 DescriptorType;

	//Algorithm params
	float model_ss_;
	float scene_ss_;
	float rf_rad_;
	float descr_rad_;
	float cg_size_;
	float cg_thresh_;

	//Results
	pcl::PointCloud<PointType>::Ptr model_keypoints;
	pcl::PointCloud<PointType>::Ptr scene_keypoints;
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
	std::vector<pcl::Correspondences> clustered_corrs;

public:
	Recognition();
	static float computeCloudResolution(const pcl::PointCloud<PointType>::ConstPtr &cloud);
	void recognize(const pcl::PointCloud<PointType>::ConstPtr &model, const pcl::PointCloud<PointType>::ConstPtr &scene);
	pcl::PointCloud<PointType>::Ptr getModelKeypoints();
	pcl::PointCloud<PointType>::Ptr getSceneKeypoints();
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > getRototranslations();
	std::vector<pcl::Correspondences> getClusteredCorrs();
	//void run(char* modelFileName, char* sceneFileName);
};

#endif
