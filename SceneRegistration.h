#ifndef REGISTRATION_H
#define REGISTRATION_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <Windows.h>
#include "TsdfVolume.cuh"
#include "RealsenseGrabber.h"

class SceneRegistration {
private:
	static std::vector<std::vector<float> > getDepth(RealsenseGrabber* grabber);
public:
	static void align(int cameras, RealsenseGrabber* grabber, Transformation* colorTrans, int targetId);
	static void align(int cameras, RealsenseGrabber* grabber, Transformation* colorTrans);
};

#endif
