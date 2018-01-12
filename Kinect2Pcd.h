#ifndef KINECT_2_PCD
#define KINECT_2_PCD

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/grabber.h>
#include "Kinect2Grabber.h"

class Kinect2Pcd {
private:
	boost::mutex mutex;
	boost::shared_ptr<pcl::Kinect2Grabber> grabber;
	boost::signals2::connection connection;

	//Result
	bool updated;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

public:
	Kinect2Pcd();
	~Kinect2Pcd();
	bool isUpdated();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPointCloud();
};

#endif
