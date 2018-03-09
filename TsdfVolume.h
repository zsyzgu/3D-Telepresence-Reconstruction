#ifndef TSDF_VOLUME_H
#define TSDF_VOLUME_H

#include <pcl/point_types.h>
#include <Windows.h>
#include <pcl/PolygonMesh.h>
#include <pcl/point_cloud.h>

class TsdfVolume {
public:
	TsdfVolume(int resolutionX, int resolutionY, int resolutionZ, float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ);
	~TsdfVolume();
	void clear();
	void integrate(UINT16* depth, RGBQUAD* color, Eigen::Matrix4f transformation);
	pcl::PolygonMesh::Ptr calnMesh();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr calnPointCloud();
};

#endif
