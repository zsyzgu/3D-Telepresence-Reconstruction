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
private:
	float3 rotation0;
	float3 rotation1;
	float3 rotation2;
	float3 translation;
public:
	CUDA_CALLABLE_MEMBER Transformation() {
		setIdentity();
	}

	CUDA_CALLABLE_MEMBER Transformation(float* rotation, float* translation) {
		rotation0 = make_float3(rotation[0], rotation[3], rotation[6]);
		rotation1 = make_float3(rotation[1], rotation[4], rotation[7]);
		rotation2 = make_float3(rotation[2], rotation[5], rotation[8]);
		this->translation = make_float3(translation[0], translation[1], translation[2]);
	}

	CUDA_CALLABLE_MEMBER Transformation(double* rotation, double* translation) {
		rotation0 = make_float3(rotation[0], rotation[1], rotation[2]);
		rotation1 = make_float3(rotation[3], rotation[4], rotation[5]);
		rotation2 = make_float3(rotation[6], rotation[7], rotation[8]);
		this->translation = make_float3(translation[0], translation[1], translation[2]);
	}

	CUDA_CALLABLE_MEMBER float3 translate(float3 pos) {
		return make_float3(dot(pos, rotation0), dot(pos, rotation1), dot(pos, rotation2)) + translation;
	}

	CUDA_CALLABLE_MEMBER float3 deltaZ() {
		return make_float3(rotation0.z, rotation1.z, rotation2.z);
	}

	CUDA_CALLABLE_MEMBER float3 col(int i) {
		if (i == 0) {
			return make_float3(rotation0.x, rotation1.x, rotation2.x);
		} else if (i == 1) {
			return make_float3(rotation0.y, rotation1.y, rotation2.y);
		} else if (i == 2) {
			return make_float3(rotation0.z, rotation1.z, rotation2.z);
		}
		return float3();
	}

	CUDA_CALLABLE_MEMBER void setIdentity() {
		rotation0 = make_float3(1, 0, 0);
		rotation1 = make_float3(0, 1, 0);
		rotation2 = make_float3(0, 0, 1);
		translation = make_float3(0, 0, 0);
	}

	CUDA_CALLABLE_MEMBER Transformation operator * (Transformation trans) {
		float3 col0 = trans.col(0);
		float3 col1 = trans.col(1);
		float3 col2 = trans.col(2);
		Transformation result;
		result.rotation0 = make_float3(dot(rotation0, col0), dot(rotation0, col1), dot(rotation0, col2));
		result.rotation1 = make_float3(dot(rotation1, col0), dot(rotation1, col1), dot(rotation1, col2));
		result.rotation2 = make_float3(dot(rotation2, col0), dot(rotation2, col1), dot(rotation2, col2));
		result.translation = make_float3(dot(rotation0, trans.translation), dot(rotation1, trans.translation), dot(rotation2, trans.translation)) + translation;
		return result;
	}
};

class Intrinsics {
public:
	float ppx, ppy;
	float fx, fy;
};

#endif
