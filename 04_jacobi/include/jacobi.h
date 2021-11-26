#ifndef _GPU_JACOBI_H_
#define _GPU_JACOBI_H_

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define MAX_ITERS 10000
#define EPS  1e-5

#include <vector>
#include "CL/cl.h"
#include "utils.h"

void jacobi_cl(float* a, float* b, float* x0, float* x1, float* norm, int size, cl_device_type device_type,
	std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time, cl_ulong& kernel_time);

#endif // _GPU_JACOBI_H_
