#ifndef PARAMETERS_H
#define PARAMETERS_H

// Compile Options
//#define CREATE_EXE
#define TRANSMISSION
#define IS_SERVER true
#define CALIBRATION false
// Camera Parameters
#define MAX_CAMERAS 8
#if CALIBRATION == false
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
// CUDA Parameters
#define BLOCK_SIZE 16
#define VOLUME 256
#define MAX_VERTEX 1000000
// Transmission
#define MAX_DELAY_FRAME 20
#define FRAME_BUFFER_SIZE 30000000
#define BUFF_SIZE 16384

#endif
