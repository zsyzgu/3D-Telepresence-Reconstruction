#ifndef TELE_CP_H
#define TELE_CP_H

#include "RealsenseGrabber.h"
#include "TsdfVolume.h"
#include "Calibration.h"
#include "Transmission.h"
#include "Parameters.h"

class TeleCP {
	/*
	Parameters:
	@grabber: Acquire RGB and depth images from realsense. RGB images are aligned to the depth camera coordinate.
	@volume: The TSDF structure. It records the data of each frame (volume->buffer).
	@calibration: Calibrate the transmission between cameras. It records the extrinsics between cameras.
	@transmission: Connect two capture sites for the teleprense.
	Functions:
	@TeleCP(): Initialization; start Realsense.
	@TeleCP(isServer): Initialization; start Realsense and TCP link.
	* TeleCP()		: Local Mode
	* TeleCP(true)	: Server Mode 
	* TeleCP(false) : client Mode
	@~TeleCP(): Clean everything; stop Realsense cameras and the TCP links.
	@update(): Acquire RGBD images from Realsense and fuse them into a 3D model. Also recv/Send data if the transmission is enabled.
	@align(): Align camera(0) with all the other cameras.
	@align(targetId): Align camera(0) with camera(targetId).
	@setOrigin(): Set the original point of the coordinate to the position of the checkerboard.
	@saveBackground(): Set the currenct 3D model as the background to be removed.
	@getBuffer(): Get the data of the current frame from TsdfVolume.
	@getPointCloud(): Get the point cloud for the PCL rendering.
	*/
private:
	RealsenseGrabber* grabber = NULL;
	TsdfVolume* volume = NULL;
	Calibration* calibration = NULL;
	Transmission* transmission = NULL;
public:
	TeleCP();
	TeleCP(bool isServer);
	~TeleCP();
	void update();
	void align();
	void align(int targetId);
	void setOrigin();
	void saveBackground();
	byte* getBuffer() { return volume->getBuffer(); }
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPointCloud() { return volume->getPointCloud(); }
};

#endif
