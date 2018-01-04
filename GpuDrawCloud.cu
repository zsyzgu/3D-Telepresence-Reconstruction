#include "GpuDrawCloud.h"

__global__ void change_points(pcl::gpu::PtrSz<pcl::PointXYZRGB> cloud_device)
{
	cloud_device[0].x += 1;
	pcl::PointXYZRGB q = cloud_device.data[0];
	printf("x=%f, y=%f, z=%f, r=%d, g=%d, b=%d \n", q.x, q.y, q.z, q.r, q.g, q.b);
}

extern "C" bool
cloud2GPU(pcl::gpu::DeviceArray<pcl::PointXYZRGB>& cloud_device)
{
	change_points << <1, 1 >> >(cloud_device);
	return true;
}
