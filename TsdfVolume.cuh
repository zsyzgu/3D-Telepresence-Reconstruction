#ifndef TSDF_VOLUME_CUH
#define TSDF_VOLUME_CUH

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include "CudaHandleError.h"
#include "Parameters.h"

#ifdef __CUDACC__
namespace tsdf {
	texture<UINT16, 3, cudaReadModeElementType> depthTexture;
	texture<UINT16, 2, cudaReadModeElementType> depthTexture0;
	texture<UINT16, 2, cudaReadModeElementType> depthTexture1;
	texture<UINT16, 2, cudaReadModeElementType> depthTexture2;
	texture<UINT16, 2, cudaReadModeElementType> depthTexture3;
	texture<UINT16, 2, cudaReadModeElementType> depthTexture4;
	texture<UINT16, 2, cudaReadModeElementType> depthTexture5;
	texture<UINT16, 2, cudaReadModeElementType> depthTexture6;
	texture<UINT16, 2, cudaReadModeElementType> depthTexture7;
	texture<uchar4, 2, cudaReadModeElementType> colorTexture0;
	texture<uchar4, 2, cudaReadModeElementType> colorTexture1;
	texture<uchar4, 2, cudaReadModeElementType> colorTexture2;
	texture<uchar4, 2, cudaReadModeElementType> colorTexture3;
	texture<uchar4, 2, cudaReadModeElementType> colorTexture4;
	texture<uchar4, 2, cudaReadModeElementType> colorTexture5;
	texture<uchar4, 2, cudaReadModeElementType> colorTexture6;
	texture<uchar4, 2, cudaReadModeElementType> colorTexture7;
}
using namespace tsdf;

__device__ texture<UINT16, 2, cudaReadModeElementType> getDepthTexture(int id) {
	if (id < 4) {
		if (id == 0) return depthTexture0;
		if (id == 1) return depthTexture1;
		if (id == 2) return depthTexture2;
		if (id == 3) return depthTexture3;
	}
	else {
		if (id == 4) return depthTexture4;
		if (id == 5) return depthTexture5;
		if (id == 6) return depthTexture6;
		if (id == 7) return depthTexture7;
	}
	return depthTexture0;
}

__device__ texture<uchar4, 2, cudaReadModeElementType> getColorTexture(int id) {
	if (id < 4) {
		if (id == 0) return colorTexture0;
		if (id == 1) return colorTexture1;
		if (id == 2) return colorTexture2;
		if (id == 3) return colorTexture3;
	}
	else {
		if (id == 4) return colorTexture4;
		if (id == 5) return colorTexture5;
		if (id == 6) return colorTexture6;
		if (id == 7) return colorTexture7;
	}
	return colorTexture0;
}

__host__ texture<UINT16, 2, cudaReadModeElementType>* getDepthTexturePointer(int id) {
	if (id < 4) {
		if (id == 0) return &depthTexture0;
		if (id == 1) return &depthTexture1;
		if (id == 2) return &depthTexture2;
		if (id == 3) return &depthTexture3;
	}
	else {
		if (id == 4) return &depthTexture4;
		if (id == 5) return &depthTexture5;
		if (id == 6) return &depthTexture6;
		if (id == 7) return &depthTexture7;
	}
	return &depthTexture0;
}

__host__ texture<uchar4, 2, cudaReadModeElementType>* getColorTexturePointer(int id) {
	if (id < 4) {
		if (id == 0) return &colorTexture0;
		if (id == 1) return &colorTexture1;
		if (id == 2) return &colorTexture2;
		if (id == 3) return &colorTexture3;
	}
	else {
		if (id == 4) return &colorTexture4;
		if (id == 5) return &colorTexture5;
		if (id == 6) return &colorTexture6;
		if (id == 7) return &colorTexture7;
	}
	return &colorTexture0;
}

#endif 

CUDA_CALLABLE_MEMBER __forceinline__ float3 operator * (float3 a, float3 b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

CUDA_CALLABLE_MEMBER __forceinline__ float3 operator + (float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

CUDA_CALLABLE_MEMBER __forceinline__ float dot(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

class Transformation {
public:
	float3 rotation0;
	float3 rotation1;
	float3 rotation2;
	float3 shift;
public:
	CUDA_CALLABLE_MEMBER Transformation() {
		rotation0 = float3();
		rotation1 = float3();
		rotation2 = float3();
		shift = float3();
	}

	CUDA_CALLABLE_MEMBER Transformation(float* matrix) {
		rotation0 = make_float3(matrix[0], matrix[1], matrix[2]);
		rotation1 = make_float3(matrix[4], matrix[5], matrix[6]);
		rotation2 = make_float3(matrix[8], matrix[9], matrix[10]);
		shift = make_float3(matrix[12], matrix[13], matrix[14]);
	}

	CUDA_CALLABLE_MEMBER Transformation(float* rotation, float* translation) {
		rotation0 = make_float3(rotation[0], rotation[3], rotation[6]);
		rotation1 = make_float3(rotation[1], rotation[4], rotation[7]);
		rotation2 = make_float3(rotation[2], rotation[5], rotation[8]);
		shift = make_float3(translation[0], translation[1], translation[2]);
	}

	CUDA_CALLABLE_MEMBER float3 translate(float3 pos) {
		return make_float3(dot(pos, rotation0), dot(pos, rotation1), dot(pos, rotation2)) + shift;
	}

	CUDA_CALLABLE_MEMBER float3 deltaZ() {
		return make_float3(rotation0.z, rotation1.z, rotation2.z);
	}

	CUDA_CALLABLE_MEMBER void setIdentity() {
		rotation0 = make_float3(1, 0, 0);
		rotation1 = make_float3(0, 1, 0);
		rotation2 = make_float3(0, 0, 1);
		shift = make_float3(0, 0, 0);
	}
};

class Intrinsics {
public:
	float ppx, ppy;
	float fx, fy;
};

#endif