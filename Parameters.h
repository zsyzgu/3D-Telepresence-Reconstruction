#ifndef PARAMETERS_H
#define PARAMETERS_H

#define MAX_CAMERAS 8
#define CALIBRATION false
#if CALIBRATION == false
	#define DEPTH_W 640
	#define DEPTH_H 480
	#define COLOR_W 960
	#define COLOR_H 540
	#define CAMERA_FPS 60
#else
	#define DEPTH_W 640
	#define DEPTH_H 480
	#define COLOR_W 1920
	#define COLOR_H 1080
	#define CAMERA_FPS 15
#endif
//CUDA Parameters
#define BLOCK_SIZE 16
#define VOLUME 256
#define MAX_VERTEX 1000000
//TRANSMISSION
#define MAX_DELAY_FRAME 20
#define FRAME_BUFFER_SIZE 30000000
#define BUFF_SIZE 16384

#endif
