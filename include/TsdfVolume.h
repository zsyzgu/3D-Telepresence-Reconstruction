#ifndef TSDF_VOLUME_H
#define TSDF_VOLUME_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Windows.h>
#include "RealsenseGrabber.h"
#include "Vertex.h"
#include "TsdfVolume.cuh"

class TsdfVolume {
	/*
	@buffer: The memory to store the 3D reconstruction. The first 4 bytes is an integer of vertex number, followed by all the vertexs stored as "Vertex" class.
	*/
private:
	byte* buffer;
public:
	TsdfVolume(float sizeX, float sizeY, float sizeZ, float centerX, float centerY, float centerZ);
	~TsdfVolume();
	void integrate(RealsenseGrabber* grabber, int remoteCameras, Transformation* world2depth);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPointCloud();
	byte* getBuffer() { return buffer; }
};

#endif
