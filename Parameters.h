#ifndef PARAMETERS_H
#define PARAMETERS_H

#define DEPTH_W 320
#define DEPTH_H 240
#define COLOR_W 640
#define COLOR_H 480
#define COLOR_FX 616.833
#define COLOR_FY -616.736
#define COLOR_CX 305.318
#define COLOR_CY 230.103
//#define DEPTH_FX 314.367 // These are origin intrinsics of DEPTH
//#define DEPTH_FY -314.367
//#define DEPTH_CX 160.593
//#define DEPTH_CY 118.191
#define DEPTH_FX 308.416 // DEPTH was aligned to COLOR, DEPTH intrinsics = 1/2 COLOR intrinsics
#define DEPTH_FY -308.368
#define DEPTH_CX 152.659
#define DEPTH_CY 115.051

#endif
