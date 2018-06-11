#ifndef PARAMETERS_H
#define PARAMETERS_H

#define MAX_CAMERAS 8
#define DEPTH_W 320
#define DEPTH_H 240
#define COLOR_W 640
#define COLOR_H 480
//CUDA Parameters
#define BLOCK_SIZE 16
#define BLOCK_SIZE_LOG 4
#define VOLUME_X 512
#define VOLUME_Y 512
#define VOLUME_Z 512
#define VOLUME_X_LOG 9
#define VOLUME_Y_LOG 9
#define VOLUME_Z_LOG 9

#endif

/*
#define DEPTH_FX 308.416 // DEPTH was aligned to COLOR, DEPTH intrinsics = 1/2 COLOR intrinsics
#define DEPTH_FY -308.368
#define DEPTH_CX 152.659
#define DEPTH_CY 115.051
*/
