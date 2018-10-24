#include "TeleCP.h"
#include <pcl/visualization/cloud_viewer.h>

TeleCP* telecp = NULL;

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event) {
	char cmd = event.getKeySym()[0];
	if (cmd == 'r' && event.keyDown()) {
		telecp->align();
	}
	if (cmd == 'o' && event.keyDown()) {
		telecp->setOrigin();
	}
	if (cmd == 'b' && event.keyDown()) {
		telecp->saveBackground();
	}
	if (cmd == '1' && event.keyUp()) {
		telecp->align(1);
	}
	if (cmd == '2' && event.keyUp()) {
		telecp->align(2);
	}
}

int main(int argc, char *argv[]) {
	telecp = new TeleCP();

	pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
	viewer.setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer.registerKeyboardCallback(keyboardEventOccurred);

	while (!viewer.wasStopped()) {
		viewer.spinOnce();

		telecp->update();

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = telecp->getPointCloud();
		if (!viewer.updatePointCloud(cloud, "cloud")) {
			viewer.addPointCloud(cloud, "cloud");
		}
	}

	delete telecp;
	return 0;
}
