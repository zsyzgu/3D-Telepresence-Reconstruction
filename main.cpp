#include "Timer.h"
#include "Recognition.h"
#include "Kinect2Pcd.h"

int main(int argc, char *argv[]) {
	Recognition recognition;
	recognition.run("save2.pcd", "save.pcd");

	//Kinect2Pcd kinect2Pcd;
	//kinect2Pcd.run();

	return 0;
}
