#ifndef TSDF_VOLUME_H
#define TSDF_VOLUME_H

#include <pcl/point_types.h>
#include <Windows.h>
#include <pcl/PolygonMesh.h>
#include <pcl/point_cloud.h>
#include "Vertex.h"

class TsdfVolume {
public:
	TsdfVolume(float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ);
	~TsdfVolume();
	void integrate(int cameras, UINT16** depth, RGBQUAD** color, Eigen::Matrix4f* transformation);
	void calnMesh(byte* buffer);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPointCloudFromMesh(byte* buffer);
};

#endif
