#ifndef VERTEX_H

#include "cuda_runtime.h"

struct Vertex {
	float3 pos;
	uchar4 color;
	uchar4 color2;
};

#endif
