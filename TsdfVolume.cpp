#include "TsdfVolume.h"
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>

extern "C" void cudaInitVolume(int resolutionX, int resolutionY, int resolutionZ, float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ);
extern "C" void cudaReleaseVolume();
extern "C" void cudaClearVolume();
extern "C" void cudaIntegrateDepth(UINT16* depth, RGBQUAD* color, float* transformation);
extern "C" void cudaCalculateMesh(float*& tris, UINT8*& tris_color, int& size);

TsdfVolume::TsdfVolume(int resolutionX, int resolutionY, int resolutionZ, float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ)
{
	cudaInitVolume(resolutionX, resolutionY, resolutionZ, sizeX, sizeY, sizeZ, centerX, centerY, centerZ);
}

TsdfVolume::~TsdfVolume()
{
	cudaReleaseVolume();
}

void TsdfVolume::clear()
{
	cudaClearVolume();
}

void TsdfVolume::integrate(UINT16 * depth, RGBQUAD* color, Eigen::Matrix4f transformation)
{
	float* trans = new float[16];
	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			trans[y * 4 + x] = transformation(y, x);
		}
	}
	cudaIntegrateDepth(depth, color, trans);
	delete[] trans;
}

pcl::PolygonMesh::Ptr TsdfVolume::calnMesh()
{
	pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());

	float* tris;
	UINT8* tris_color;
	int size;
	cudaCalculateMesh(tris, tris_color, size);

	pcl::PointCloud<pcl::PointXYZRGB> cloud;
	cloud.resize(size * 3);
	cloud.width = size * 3;
	cloud.height = 1;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < 3; j++) {
			cloud.points[i * 3 + j].x = tris[i * 9 + j * 3 + 0];
			cloud.points[i * 3 + j].y = tris[i * 9 + j * 3 + 1];
			cloud.points[i * 3 + j].z = tris[i * 9 + j * 3 + 2];
			cloud.points[i * 3 + j].r = tris_color[i * 9 + j * 3 + 0];
			cloud.points[i * 3 + j].g = tris_color[i * 9 + j * 3 + 1];
			cloud.points[i * 3 + j].b = tris_color[i * 9 + j * 3 + 2];
		}
	}
	pcl::toPCLPointCloud2(cloud, mesh->cloud);

	mesh->polygons.resize(size);
	for (int i = 0; i < size; i++) {
		pcl::Vertices v;
		v.vertices.push_back(i * 3 + 0);
		v.vertices.push_back(i * 3 + 2);
		v.vertices.push_back(i * 3 + 1);
		mesh->polygons[i] = v;
	}

	delete[] tris;
	delete[] tris_color;
	return mesh;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr TsdfVolume::calnPointCloud() {
	float* tris;
	UINT8* tris_color;
	int size;
	cudaCalculateMesh(tris, tris_color, size);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	cloud->resize(size * 3);
	cloud->width = size * 3;
	cloud->height = 1;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < 3; j++) {
			cloud->points[i * 3 + j].x = tris[i * 9 + j * 3 + 0];
			cloud->points[i * 3 + j].y = tris[i * 9 + j * 3 + 1];
			cloud->points[i * 3 + j].z = tris[i * 9 + j * 3 + 2];
			cloud->points[i * 3 + j].r = tris_color[i * 9 + j * 3 + 0]; //TODO
			cloud->points[i * 3 + j].g = tris_color[i * 9 + j * 3 + 1];
			cloud->points[i * 3 + j].b = tris_color[i * 9 + j * 3 + 2];
		}
	}

	return cloud;
}
