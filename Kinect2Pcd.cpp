#include "Kinect2Pcd.h"
#include "pcl/io/pcd_io.h"
#include "kinect2_grabber.h"

Kinect2Pcd::Kinect2Pcd()
{

}

void Kinect2Pcd::run()
{
	// PCL Visualizer
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
		new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.5, 0.0, 0.0, 0.0);

	// Point Cloud
	pcl::PointCloud<PointType>::ConstPtr cloud;

	// Retrieved Point Cloud Callback Function
	boost::mutex mutex;
	boost::function<void(const pcl::PointCloud<PointType>::ConstPtr&)> function =
		[&cloud, &mutex](const pcl::PointCloud<PointType>::ConstPtr& ptr) {
		boost::mutex::scoped_lock lock(mutex);

		/* Point Cloud Processing */

		cloud = ptr->makeShared();
	};

	// Kinect2Grabber
	boost::shared_ptr<pcl::Grabber> grabber = boost::make_shared<pcl::Kinect2Grabber>();

	// Register Callback Function
	boost::signals2::connection connection = grabber->registerCallback(function);

	// Start Grabber
	grabber->start();

	while (!viewer->wasStopped()) {
		// Update Viewer
		viewer->spinOnce();

		boost::mutex::scoped_try_lock lock(mutex);
		if (lock.owns_lock() && cloud) {
			// Update Point Cloud
			if (!viewer->updatePointCloud(cloud, "cloud")) {
				viewer->addPointCloud(cloud, "cloud");
			}
		}
	}

	// Stop Grabber
	grabber->stop();

	// Disconnect Callback Function
	if (connection.connected()) {
		connection.disconnect();
	}

	pcl::io::savePCDFileASCII("save.pcd", *cloud);

	pcl::PointCloud<PointType>::Ptr cloud2(new pcl::PointCloud<PointType>());
	float minX = 1e8, maxX = -1e8;
	float minY = 1e8, maxY = -1e8;
	for (int i = cloud->size() / 3; i < cloud->size() / 3 * 2; i++) {
		float x = cloud->at(i).x;
		float y = cloud->at(i).y;
		minX = std::min(minX, x);
		maxX = std::max(maxX, x);
		minY = std::min(minY, y);
		maxY = std::max(maxY, y);
	}
	for (int i = cloud->size() / 3; i < cloud->size() / 3 * 2; i++) {
		float x = cloud->at(i).x;
		float y = cloud->at(i).y;
		if (minX * 2 / 3 + maxX / 3 <= x && x <= minX / 3 + maxX * 2 / 3) {
			if (minY * 2 / 3 + maxY / 3 <= y && y <= minY / 3 + maxY * 2 / 3) {
				cloud2->push_back(cloud->at(i));
			}
		}
	}
	pcl::io::savePCDFileASCII("save2.pcd", *cloud2);
}
