/*
* gpu_draw_cloud.h
*
*  Created on: Nov 25, 2016
*      Author: lzp
*/

#ifndef INCLUDES_GPU_DRAW_CLOUD_H_
#define INCLUDES_GPU_DRAW_CLOUD_H_


#include <iostream>
#include <pcl/point_types.h>
#include <pcl/gpu/containers/device_array.h>

/*check if the compiler is of C++*/
#ifdef __cplusplus


/*
* Try accessing GPU with pointcloud
* */
extern "C" bool cloud2GPU(pcl::gpu::DeviceArray<pcl::PointXYZRGB>& cloud_device);


#endif


#endif /* INCLUDES_GPU_DRAW_CLOUD_H_ */
