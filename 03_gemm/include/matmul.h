#ifndef _GPU_MATMUL_H_
#define _GPU_MATMUL_H_

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define THREADS 6
#define BLOCK 16

#include <vector>
#include "CL/cl.h"
#include "utils.h"

void matmul(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c);
void matmul_omp(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c);
void matmul_cl(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c,
	std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time);

void gemm_cl(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c,
	std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time);
void gemm_image_cl(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c,
	std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time);

#endif _GPU_MATMUL_H_
