#ifndef _GPU_HETERO_ALGORITHM_H_
#define _GPU_HETERO_ALGORITHM_H_

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define BLOCK 16
#define MAX_ITERS 50000
#define EPS  1e-5

#include <vector>
#include "CL/cl.h"
#include "utils.h"

void matmul(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c);
void gemm_cl(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c,
	std::pair<cl_platform_id, cl_device_id>& cpu_dev_pair, std::pair<cl_platform_id, cl_device_id>& gpu_dev_pair,
	timer& time, const size_t gpu_m);

void jacobi_cl(float* a, float* b, float* x0, float* x1, float* norm, int size,
	std::pair<cl_platform_id, cl_device_id>& cpu_dev_pair, std::pair<cl_platform_id, cl_device_id>& gpu_dev_pair, timer& time,
	cl_ulong& kernel_time, const int gpu_m);


#endif // _GPU_HETERO_ALGORITHM_H_
