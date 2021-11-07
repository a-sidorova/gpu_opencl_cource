#include "../include/matmul.h"

void matmul(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c) {
	for (auto i = 0; i < m; ++i) {
		for (auto j = 0; j < k; ++j) {
			c[i * k + j] = 0;
			for (int l = 0; l < n; ++l) {
				c[i * k + j] += a[i * n + l] * b[l * k + j];
			}
		}
	}
}

void matmul_omp(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c) {
	int i, j, l;
#pragma omp parallel for private(i, j, l) num_threads(THREADS)
	for (i = 0; i < static_cast<int>(m); ++i) {
		for (j = 0; j < static_cast<int>(k); ++j) {
			c[i * k + j] = 0;
			for (l = 0; l < static_cast<int>(n); ++l) {
				c[i * k + j] += a[i * n + l] * b[l * k + j];
			}
		}
	}
}

void matmul_cl(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c,
	std::pair<cl_platform_id, cl_device_id>& dev_pair,
	std::pair<std::chrono::high_resolution_clock::time_point, std::chrono::high_resolution_clock::time_point>& time) {
    cl_int error = CL_SUCCESS;
    cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)dev_pair.first, 0 };

    cl_context context = clCreateContext(properties, 1, &dev_pair.second, nullptr, nullptr, &error);
    CONTROL("clCreateContext", error);

    cl_command_queue queue = clCreateCommandQueue(context, dev_pair.second, 0, &error);
    CONTROL("clCreateCommandQueuC", error);

    cl_program program = createProgramFromSource(context, "kernels/matmul_kernel.cl");
    CONTROL("clBuildProgram", clBuildProgram(program, 1, &dev_pair.second, nullptr, nullptr, nullptr));

    cl_kernel kernel = clCreateKernel(program, "matmul", &error);
    CONTROL("clCreateKernel", error);

    cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * m * n, nullptr, &error);
    CONTROL("clCreateBuffer A", error);
    cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n * k, nullptr, &error);
    CONTROL("clCreateBuffer B", error);
    cl_mem c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * m * k, nullptr, &error);
    CONTROL("clCreateBuffer C", error);

    CONTROL("clEnqueueWriteBuffer A", clEnqueueWriteBuffer(queue, a_buffer, CL_TRUE, 0, sizeof(float) * m * n, a, 0, nullptr, nullptr));
    CONTROL("clEnqueueWriteBuffer B", clEnqueueWriteBuffer(queue, b_buffer, CL_TRUE, 0, sizeof(float) * n * k, b, 0, nullptr, nullptr));

    CONTROL("clSetKernelArg M", clSetKernelArg(kernel, 0, sizeof(unsigned int), &m));
    CONTROL("clSetKernelArg N", clSetKernelArg(kernel, 1, sizeof(unsigned int), &n));
    CONTROL("clSetKernelArg K", clSetKernelArg(kernel, 2, sizeof(unsigned int), &k));
    CONTROL("clSetKernelArg A", clSetKernelArg(kernel, 3, sizeof(cl_mem), &a_buffer));
    CONTROL("clSetKernelArg B", clSetKernelArg(kernel, 4, sizeof(cl_mem), &b_buffer));
    CONTROL("clSetKernelArg C", clSetKernelArg(kernel, 5, sizeof(cl_mem), &c_buffer));

    size_t group = 16;
    CONTROL("clGetKernelWorkGroupInf", clGetKernelWorkGroupInfo(kernel, dev_pair.second, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr));

	size_t* global = new size_t[2];
	global[0] = m;
	global[1] = k;

	size_t* local = new size_t[2];
	local[0] = static_cast<size_t>(BLOCK);
	local[1] = static_cast<size_t>(BLOCK);

	const size_t ndims = 2;
    time.first = std::chrono::high_resolution_clock::now();
    CONTROL("clEnqueueNDRangeKernel", clEnqueueNDRangeKernel(queue, kernel, ndims, nullptr, global, local, 0, nullptr, nullptr));
    CONTROL("clFinish", clFinish(queue));
    time.second = std::chrono::high_resolution_clock::now();

    clEnqueueReadBuffer(queue, c_buffer, CL_TRUE, 0, sizeof(float) * m * k, c, 0, nullptr, nullptr);

    clReleaseMemObject(a_buffer);
    clReleaseMemObject(b_buffer);
    clReleaseMemObject(c_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void gemm_cl(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c,
    std::pair<cl_platform_id, cl_device_id>& dev_pair,
    std::pair<std::chrono::high_resolution_clock::time_point, std::chrono::high_resolution_clock::time_point>& time) {
        cl_int error = CL_SUCCESS;
        cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)dev_pair.first, 0 };

        cl_context context = clCreateContext(properties, 1, &dev_pair.second, nullptr, nullptr, &error);
        CONTROL("clCreateContext", error);

        cl_command_queue queue = clCreateCommandQueue(context, dev_pair.second, 0, &error);
        CONTROL("clCreateCommandQueuC", error);

        cl_program program = createProgramFromSource(context, "kernels/gemm_kernel.cl");
        std::string build_options = "-DBLOCK=" + std::to_string(BLOCK);
        CONTROL("clBuildProgram", clBuildProgram(program, 1, &dev_pair.second, build_options.c_str(), nullptr, nullptr));

        cl_kernel kernel = clCreateKernel(program, "gemm", &error);
        CONTROL("clCreateKernel", error);

        cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * m * n, nullptr, &error);
        CONTROL("clCreateBuffer A", error);
        cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n * k, nullptr, &error);
        CONTROL("clCreateBuffer B", error);
        cl_mem c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * m * k, nullptr, &error);
        CONTROL("clCreateBuffer C", error);

        CONTROL("clEnqueueWriteBuffer A", clEnqueueWriteBuffer(queue, a_buffer, CL_TRUE, 0, sizeof(float)* m* n, a, 0, nullptr, nullptr));
        CONTROL("clEnqueueWriteBuffer B", clEnqueueWriteBuffer(queue, b_buffer, CL_TRUE, 0, sizeof(float)* n* k, b, 0, nullptr, nullptr));

        CONTROL("clSetKernelArg M", clSetKernelArg(kernel, 0, sizeof(unsigned int), &m));
        CONTROL("clSetKernelArg N", clSetKernelArg(kernel, 1, sizeof(unsigned int), &n));
        CONTROL("clSetKernelArg K", clSetKernelArg(kernel, 2, sizeof(unsigned int), &k));
        CONTROL("clSetKernelArg A", clSetKernelArg(kernel, 3, sizeof(cl_mem), &a_buffer));
        CONTROL("clSetKernelArg B", clSetKernelArg(kernel, 4, sizeof(cl_mem), &b_buffer));
        CONTROL("clSetKernelArg C", clSetKernelArg(kernel, 5, sizeof(cl_mem), &c_buffer));

        size_t group = 16;
        CONTROL("clGetKernelWorkGroupInf", clGetKernelWorkGroupInfo(kernel, dev_pair.second, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr));

        size_t* global = new size_t[2];
        global[0] = m;
        global[1] = k;

        size_t* local = new size_t[2];
        local[0] = static_cast<size_t>(BLOCK);
        local[1] = static_cast<size_t>(BLOCK);

        const size_t ndims = 2;
        time.first = std::chrono::high_resolution_clock::now();
        CONTROL("clEnqueueNDRangeKernel", clEnqueueNDRangeKernel(queue, kernel, ndims, nullptr, global, local, 0, nullptr, nullptr));
        CONTROL("clFinish", clFinish(queue));
        time.second = std::chrono::high_resolution_clock::now();

        clEnqueueReadBuffer(queue, c_buffer, CL_TRUE, 0, sizeof(float)* m * k, c, 0, nullptr, nullptr);

        clReleaseMemObject(a_buffer);
        clReleaseMemObject(b_buffer);
        clReleaseMemObject(c_buffer);
        clReleaseProgram(program);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
}

void gemm_image_cl(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c,
    std::pair<cl_platform_id, cl_device_id>& dev_pair,
    std::pair<std::chrono::high_resolution_clock::time_point, std::chrono::high_resolution_clock::time_point>& time) {
    cl_int error = CL_SUCCESS;
    cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)dev_pair.first, 0 };

    cl_context context = clCreateContext(properties, 1, &dev_pair.second, nullptr, nullptr, &error);
    CONTROL("clCreateContext", error);

    cl_command_queue queue = clCreateCommandQueue(context, dev_pair.second, 0, &error);
    CONTROL("clCreateCommandQueuC", error);

    cl_program program = createProgramFromSource(context, "kernels/gemm_kernel.cl");
    std::string build_options = "-DBLOCK=" + std::to_string(BLOCK);
    CONTROL("clBuildProgram", clBuildProgram(program, 1, &dev_pair.second, build_options.c_str(), nullptr, nullptr));

    cl_kernel kernel = clCreateKernel(program, "gemm_image", &error);
    CONTROL("clCreateKernel", error);

    cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * m * n, nullptr, &error);
    CONTROL("clCreateBuffer A", error);
    cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n * k, nullptr, &error);
    CONTROL("clCreateBuffer B", error);
    cl_mem c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * m * k, nullptr, &error);
    CONTROL("clCreateBuffer C", error);

    CONTROL("clEnqueueWriteBuffer A", clEnqueueWriteBuffer(queue, a_buffer, CL_TRUE, 0, sizeof(float) * m * n, a, 0, nullptr, nullptr));
    CONTROL("clEnqueueWriteBuffer B", clEnqueueWriteBuffer(queue, b_buffer, CL_TRUE, 0, sizeof(float) * n * k, b, 0, nullptr, nullptr));

    CONTROL("clSetKernelArg M", clSetKernelArg(kernel, 0, sizeof(unsigned int), &m));
    CONTROL("clSetKernelArg N", clSetKernelArg(kernel, 1, sizeof(unsigned int), &n));
    CONTROL("clSetKernelArg K", clSetKernelArg(kernel, 2, sizeof(unsigned int), &k));
    CONTROL("clSetKernelArg A", clSetKernelArg(kernel, 3, sizeof(cl_mem), &a_buffer));
    CONTROL("clSetKernelArg B", clSetKernelArg(kernel, 4, sizeof(cl_mem), &b_buffer));
    CONTROL("clSetKernelArg C", clSetKernelArg(kernel, 5, sizeof(cl_mem), &c_buffer));

    size_t group = 16;
    CONTROL("clGetKernelWorkGroupInf", clGetKernelWorkGroupInfo(kernel, dev_pair.second, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr));

    size_t* global = new size_t[2];
    global[0] = m;
    global[1] = k;

    size_t* local = new size_t[2];
    local[0] = static_cast<size_t>(BLOCK);
    local[1] = static_cast<size_t>(BLOCK);

    const size_t ndims = 2;
    time.first = std::chrono::high_resolution_clock::now();
    CONTROL("clEnqueueNDRangeKernel", clEnqueueNDRangeKernel(queue, kernel, ndims, nullptr, global, local, 0, nullptr, nullptr));
    CONTROL("clFinish", clFinish(queue));
    time.second = std::chrono::high_resolution_clock::now();

    clEnqueueReadBuffer(queue, c_buffer, CL_TRUE, 0, sizeof(float) * m * k, c, 0, nullptr, nullptr);

    clReleaseMemObject(a_buffer);
    clReleaseMemObject(b_buffer);
    clReleaseMemObject(c_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
