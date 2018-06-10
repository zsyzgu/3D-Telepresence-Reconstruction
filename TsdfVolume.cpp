#include "TsdfVolume.h"
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>
#include "Timer.h"

extern "C" void cudaInitVolume(float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ);
extern "C" void cudaReleaseVolume();
extern "C" void cudaIntegrateDepth(int cameras, UINT16** depth, RGBQUAD** color, float** transformation);
extern "C" void cudaCalculateMesh(Vertex* vertex, int& size);

TsdfVolume::TsdfVolume(float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ)
{
	cudaInitVolume(sizeX, sizeY, sizeZ, centerX, centerY, centerZ);
}

TsdfVolume::~TsdfVolume()
{
	cudaReleaseVolume();
}

void TsdfVolume::integrate(int cameras, UINT16** depth, RGBQUAD** color, Eigen::Matrix4f* transformation)
{
	float** trans = new float*[cameras];
	for (int i = 0; i < cameras; i++) {
		trans[i] = transformation[i].data();
	}

	cudaIntegrateDepth(cameras, depth, color, trans);

	delete[] trans;
}

void TsdfVolume::calnMesh(byte* buffer)
{
	Vertex* vertex = (Vertex*)(buffer + 4);
	cudaCalculateMesh(vertex, *((int*)buffer));
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr TsdfVolume::getPointCloudFromMesh(byte* buffer)
{
	int size = *((int*)buffer);
	Vertex* vertex = (Vertex*)(buffer + 4);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	cloud->resize(size * 3);

#pragma omp parallel for schedule(static, 500)
	for (int i = 0; i < size * 3; i++) {
		cloud->points[i].x = vertex[i].pos.x;
		cloud->points[i].y = vertex[i].pos.y;
		cloud->points[i].z = vertex[i].pos.z;
		cloud->points[i].r = vertex[i].color.x;
		cloud->points[i].g = vertex[i].color.y;
		cloud->points[i].b = vertex[i].color.z;
	}

	return cloud;
}
