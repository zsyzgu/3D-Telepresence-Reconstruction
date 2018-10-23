#ifndef TSDF_VOLUME_H
#define TSDF_VOLUME_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Windows.h>
#include "RealsenseGrabber.h"
#include "Vertex.h"
#include "TsdfVolume.cuh"

class TsdfVolume {
public:
	TsdfVolume(float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ);
	~TsdfVolume();
	void integrate(byte* result, RealsenseGrabber* grabber, int remoteCameras, Transformation* world2depth);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPointCloudFromMesh(byte* buffer);
};

#endif
