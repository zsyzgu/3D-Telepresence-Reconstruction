#include "Kinect2Pcd.h"
#include "pcl/io/pcd_io.h"
#include "kinect2_grabber.h"

Kinect2Pcd::Kinect2Pcd() {
	boost::function<void(const pcl::PointCloud<PointType>::ConstPtr&)> function = [this](const pcl::PointCloud<PointType>::ConstPtr& ptr) {
		boost::mutex::scoped_lock lock(mutex);
		cloud = ptr->makeShared();
		updated = true;
	};

	grabber = boost::make_shared<pcl::Kinect2Grabber>();
	connection = grabber->registerCallback(function);

	grabber->start();
	updated = false;
}

Kinect2Pcd::~Kinect2Pcd() {
	grabber->stop();

	if (connection.connected()) {
		connection.disconnect();
	}
}

bool Kinect2Pcd::isUpdated() {
	return updated;
}

pcl::PointCloud<Kinect2Pcd::PointType>::Ptr Kinect2Pcd::getPointCloud() {
	while (true) {
		boost::mutex::scoped_try_lock lock(mutex);
		if (lock.owns_lock()) {
			break;
		}
	}

	updated = false;

	if (cloud) {
		return cloud;
	}
	else {
		return pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
	}
}
