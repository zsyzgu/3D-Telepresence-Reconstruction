#ifndef POINT_CLOUD_FILTER_H
#define POINT_CLOUD_FILTER_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

class PointCloudProcess {
public:
	static void mlsFiltering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
	static void merge2PointClouds(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud1, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud2);
	static void pointCloud2Mesh(pcl::PolygonMesh::Ptr mesh, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
};

#endif
