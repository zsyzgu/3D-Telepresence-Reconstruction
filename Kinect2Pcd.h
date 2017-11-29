#ifndef KINECT_2_PCD
#define KINECT_2_PCD

#include <pcl/visualization/pcl_visualizer.h>

class Kinect2Pcd {
private:
	typedef pcl::PointXYZRGBA PointType;
public:
	Kinect2Pcd();
	void run();
};

#endif
