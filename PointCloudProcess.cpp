#include "PointCloudProcess.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
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
#include "pcl/surface/gp3.h"
#include "Timer.h"

void PointCloudProcess::mlsFiltering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGBNormal> mlsPoints;
	pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> mls;
	mls.setComputeNormals(true);
	mls.setInputCloud(cloud);
	mls.setPolynomialFit(true);
	mls.setSearchMethod(tree);
	mls.setSearchRadius(0.01);
	mls.setPolynomialOrder(1);


	Timer timer;
	timer.reset();
	mls.process(mlsPoints);
	std::cout << timer.getTime() * 1e3f << " ms" << std::endl;


	cloud->points.resize(mlsPoints.size());
	for (int i = 0; i < cloud->size(); i++) {
		cloud->points[i].x = mlsPoints.points[i].x;
		cloud->points[i].y = mlsPoints.points[i].y;
		cloud->points[i].z = mlsPoints.points[i].z;
		cloud->points[i].r = mlsPoints.points[i].r;
		cloud->points[i].g = mlsPoints.points[i].g;
		cloud->points[i].b = mlsPoints.points[i].b;
	}
	cloud->width = cloud->size();
	cloud->height = 1;
}

void PointCloudProcess::merge2PointClouds(pcl::PointCloud<pcl::PointXYZRGB>::Ptr model, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr model1, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr model2)
{
	float resolution = 0.0025; //(Recognition::computeCloudResolution(model1) + Recognition::computeCloudResolution(model2)) / 2;

	pcl::search::KdTree<pcl::PointXYZRGB> tree;
	tree.setInputCloud(model2);

	std::vector<int> indices(1);
	std::vector<float> sqrDistances(2);
	std::vector<bool> usedModel2Point(model2->size());
	for (int i = 0; i < model1->size(); i++) {
		pcl::PointXYZRGB point = model1->at(i);
		if (!pcl_isfinite(point.x)) {
			continue;
		}

		int nres = tree.nearestKSearch(point, 1, indices, sqrDistances);

		if (nres == 1) {
			int index = indices[0];
			float distance = sqrt(sqrDistances[0]);

			if (distance < resolution) {
				pcl::PointXYZRGB point2 = model2->at(index);
				pcl::PointXYZRGB avePoint(((UINT16)point.r + point2.r) / 2, ((UINT16)point.g + point2.g) / 2, ((UINT16)point.g + point2.g) / 2);
				avePoint.x = (point.x + point2.x) / 2;
				avePoint.y = (point.y + point2.y) / 2;
				avePoint.z = (point.z + point2.z) / 2;
				model->push_back(avePoint);
				usedModel2Point[index] = true;
			}
			else {
				model->push_back(point);
			}
		}
	}
	for (int i = 0; i < model2->size(); i++) {
		pcl::PointXYZRGB point = model2->at(i);
		if (!pcl_isfinite(point.x)) {
			continue;
		}
		if (!usedModel2Point[i]) {
			model->push_back(point);
		}
	}
}

void PointCloudProcess::pointCloud2Mesh(pcl::PolygonMesh::Ptr mesh, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normalEstimation;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	kdTree->setInputCloud(cloud);
	normalEstimation.setInputCloud(cloud);
	normalEstimation.setSearchMethod(kdTree);
	normalEstimation.setKSearch(10);
	normalEstimation.compute(*normals);

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudWithNormals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloudWithNormals);

	pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
	tree2->setInputCloud(cloudWithNormals);

	pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;

	gp3.setSearchRadius(0.025);

	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(50);
	gp3.setMaximumSurfaceAngle(M_PI / 4);
	gp3.setMinimumAngle(M_PI / 18);
	gp3.setMaximumAngle(M_PI * 2 / 3);
	gp3.setNormalConsistency(false);

	gp3.setInputCloud(cloudWithNormals);
	gp3.setSearchMethod(tree2);
	gp3.reconstruct(*mesh);
}

void PointCloudProcess::pointCloud2PCNormal(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pcNormal, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	mlsFiltering(cloud);

	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normalEstimation;

	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	kdTree->setInputCloud(cloud);
	normalEstimation.setInputCloud(cloud);
	normalEstimation.setSearchMethod(kdTree);
	normalEstimation.setKSearch(20);
	normalEstimation.compute(*normals);

	pcl::concatenateFields(*cloud, *normals, *pcNormal);
}
