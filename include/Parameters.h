#ifndef PARAMETERS_H
#define PARAMETERS_H

#define HD false
#define MAX_CAMERAS 8
#if HD == false
	#define DEPTH_W 640
	#define DEPTH_H 480
	#define COLOR_W 960
	#define COLOR_H 540
	#define CAMERA_FPS 30
#else
	#define DEPTH_W 640
	#define DEPTH_H 480
	#define COLOR_W 1920
	#define COLOR_H 1080
	#define CAMERA_FPS 30
#endif

#endif
