#include "Kinect2Pcd.h"
#include "Kinect2Grabber.h"
#include "pcl/io/pcd_io.h"

Kinect2Pcd::Kinect2Pcd() {
	cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());

	boost::function<void(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&)> function = [this](const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& ptr) {
		boost::mutex::scoped_lock lock(mutex);
		cloud = ptr->makeShared();
		updated = true;
	};

	grabber = boost::make_shared<pcl::Kinect2Grabber>();
//	connection = grabber->registerCallback(function);

	grabber->start();
	updated = false;
}

Kinect2Pcd::~Kinect2Pcd() {
//	grabber->stop();

	if (connection.connected()) {
		connection.disconnect();
	}
}

bool Kinect2Pcd::isUpdated() {
	return updated;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Kinect2Pcd::getPointCloud() {
	while (true) {
		boost::mutex::scoped_try_lock lock(mutex);
		if (lock.owns_lock()) {
			break;
		}
	}

	updated = false;
	return cloud;
}
