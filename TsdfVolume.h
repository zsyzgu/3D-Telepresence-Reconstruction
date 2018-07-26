#ifndef TSDF_VOLUME_H
#define TSDF_VOLUME_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Windows.h>
#include "Vertex.h"
#include "TsdfVolume.cuh"

class TsdfVolume {
public:
	TsdfVolume(float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ);
	~TsdfVolume();
	void integrate(byte* result, int cameras, float* depth_device, RGBQUAD* color_device, Transformation* world2depth, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPointCloudFromMesh(byte* buffer);
};

#endif
