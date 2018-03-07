#include "TsdfVolume.h"
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>

extern "C" void cudaInitVolume(int resolutionX, int resolutionY, int resolutionZ, float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ);
extern "C" void cudaReleaseVolume();
extern "C" void cudaClearVolume();
extern "C" void cudaIntegrateDepth(UINT16* depth, float* transformation);
extern "C" void cudaCalculateMesh(float*& tris, int& size);

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

void TsdfVolume::integrate(UINT16 * depth, Eigen::Matrix4f transformation)
{
	float* trans = new float[16];
	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			trans[y * 4 + x] = transformation(y, x);
		}
	}
	cudaIntegrateDepth(depth, trans);
	delete[] trans;
}

pcl::PolygonMesh::Ptr TsdfVolume::calnMesh()
{
	pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());

	float* tris;
	int size;
	cudaCalculateMesh(tris, size);

	pcl::PointCloud<pcl::PointXYZ> cloud;
	cloud.resize(size * 3);
	cloud.width = size * 3;
	cloud.height = 1;
	for (int i = 0; i < size; i++) {
		cloud.points[i * 3 + 0].x = tris[i * 9 + 0];
		cloud.points[i * 3 + 0].y = tris[i * 9 + 1];
		cloud.points[i * 3 + 0].z = tris[i * 9 + 2];
		cloud.points[i * 3 + 1].x = tris[i * 9 + 3];
		cloud.points[i * 3 + 1].y = tris[i * 9 + 4];
		cloud.points[i * 3 + 1].z = tris[i * 9 + 5];
		cloud.points[i * 3 + 2].x = tris[i * 9 + 6];
		cloud.points[i * 3 + 2].y = tris[i * 9 + 7];
		cloud.points[i * 3 + 2].z = tris[i * 9 + 8];
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
	return mesh;
}
