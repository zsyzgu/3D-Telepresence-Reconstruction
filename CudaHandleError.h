#ifndef CUDA_HANDLE_ERROR_H
#define CUDA_HANDLE_ERROR_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "stdio.h"
#include <Windows.h>
#include <iostream>

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif
