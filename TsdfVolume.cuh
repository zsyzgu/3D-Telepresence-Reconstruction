#ifndef TSDF_VOLUME_CUH
#define TSDF_VOLUME_CUH

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include "CudaHandleError.h"
#include "Parameters.h"

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
		rotation0 = make_float3(rotation[0], rotation[1], rotation[2]);
		rotation1 = make_float3(rotation[3], rotation[4], rotation[5]);
		rotation2 = make_float3(rotation[6], rotation[7], rotation[8]);
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
