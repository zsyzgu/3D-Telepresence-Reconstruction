#include "TsdfVolume.h"
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>
#include "Timer.h"

extern "C" void cudaInitVolume(float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ);
extern "C" void cudaReleaseVolume();
extern "C" void cudaIntegrate(int cameras, RealsenseGrabber* grabber, int localCameras, int& triSize, Vertex* vertex, Transformation* world2depth);

TsdfVolume::TsdfVolume(float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ)
{
	cudaInitVolume(sizeX, sizeY, sizeZ, centerX, centerY, centerZ);
}

TsdfVolume::~TsdfVolume()
{
	cudaReleaseVolume();
}

void TsdfVolume::integrate(byte* result, RealsenseGrabber* grabber, int cameras, int localCameras, Transformation* world2depth)
{
	Vertex* vertex = (Vertex*)(result + 4);
	cudaIntegrate(cameras, grabber, localCameras, *((int*)result), vertex, world2depth);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr TsdfVolume::getPointCloudFromMesh(byte* buffer)
{
	int size = *((int*)buffer);
	Vertex* vertex = (Vertex*)(buffer + 4);

	int n = size * 3;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	cloud->resize(n * 2);

	for (int i = 0; i < n; i++) {
		cloud->points[i].x = vertex[i].pos.x;
		cloud->points[i].y = vertex[i].pos.y;
		cloud->points[i].z = vertex[i].pos.z;
		cloud->points[i].r = vertex[i].color.x;
		cloud->points[i].g = vertex[i].color.y;
		cloud->points[i].b = vertex[i].color.z;
		int j = i + 1;
		if (j % 3 == 0) {
			j -= 3;
		}
		cloud->points[n + i].x = (vertex[i].pos.x + vertex[j].pos.x) * 0.5;
		cloud->points[n + i].y = (vertex[i].pos.y + vertex[j].pos.y) * 0.5;
		cloud->points[n + i].z = (vertex[i].pos.z + vertex[j].pos.z) * 0.5;
		cloud->points[n + i].r = vertex[i].color2.x;
		cloud->points[n + i].g = vertex[i].color2.y;
		cloud->points[n + i].b = vertex[i].color2.z;
	}

	return cloud;
}
