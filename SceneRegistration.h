#ifndef REGISTRATION_H
#define REGISTRATION_H

#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/shot_omp.h>

class SceneRegistration {
public:
	SceneRegistration();
	~SceneRegistration();

public:
	Eigen::Matrix4f align(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr source, pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr target);
};

#endif
