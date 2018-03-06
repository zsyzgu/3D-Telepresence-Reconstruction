#include "TsdfVolume.h"

extern "C" void cudaInitVolume(int resolutionX, int resolutionY, int resolutionZ, float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ);
extern "C" void cudaReleaseVolume();
extern "C" void cudaClearVolume();
extern "C" void cudaIntegrateDepth(UINT16* depth, float* transformation);
extern "C" void cudaCalculateMesh(float* tris, int& size);

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

pcl::PolygonMesh TsdfVolume::calnMesh()
{
	pcl::PolygonMesh mesh;

	float* tris;
	int size;
	cudaCalculateMesh(tris, size);

	return mesh;
}
