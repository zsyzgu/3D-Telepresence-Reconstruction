#ifndef REGISTRATION_H
#define REGISTRATION_H

#include "pcl/registration/icp.h"
#include "pcl/kdtree/kdtree.h"
#include "pcl/registration/ia_ransac.h"
#include "pcl/features/shot_omp.h"

class SceneRegistration : public pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float> {
public:
	SceneRegistration();
	~SceneRegistration();

	void transform(const Eigen::Matrix4f& mat, const pcl::PointXYZ& p, pcl::PointXYZ& out);

protected:
	static inline bool compareCorrespondences(const pcl::Correspondence& a, const pcl::Correspondence& b) {
		return a.distance < b.distance;
	}

public:
	Eigen::Matrix4f align(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr source, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr target);
};

#endif
