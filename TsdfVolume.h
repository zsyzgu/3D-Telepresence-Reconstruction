#ifndef TSDF_VOLUME_H
#define TSDF_VOLUME_H

#include <pcl/point_types.h>
#include <Windows.h>
#include <pcl/PolygonMesh.h>
#include <pcl/point_cloud.h>
#include "Vertex.h"
#include "TsdfVolume.cuh"

class TsdfVolume {
public:
	TsdfVolume(float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ);
	~TsdfVolume();
	void integrate(int cameras, UINT16** depth, RGBQUAD** color, Transformation* toWorldTrans, Transformation* depthToColorTrans, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics);
	void calnMesh(byte* buffer);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPointCloudFromMesh(byte* buffer);
};

#endif
