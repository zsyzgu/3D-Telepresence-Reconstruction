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

CUDA_CALLABLE_MEMBER __forceinline__ float3 operator * (float3 a, float k) {
	return make_float3(a.x * k, a.y * k, a.z * k);
}

CUDA_CALLABLE_MEMBER __forceinline__ float3 operator + (float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

CUDA_CALLABLE_MEMBER __forceinline__ float3 operator - (float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
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
	CUDA_CALLABLE_MEMBER float2 translate(float3 pos) {
		return make_float2(pos.x * fx / pos.z + ppx, pos.y * fy / pos.z + ppy);
	}
};

#endif
