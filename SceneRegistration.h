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
public:
	static Transformation align(RealsenseGrabber* grabber);
};

#endif
