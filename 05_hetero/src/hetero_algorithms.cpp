#include "..\include\hetero_algorithms.h"

void matmul(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c) {
    std::memset(c, 0, n * k * sizeof(float));

    for (auto i = 0; i < m; ++i) {
        for (int l = 0; l < n; ++l) {
            for (auto j = 0; j < k; ++j) {
                c[i * k + j] += a[i * n + l] * b[l * k + j];
            }
        }
    }
}

void gemm_cl(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c,
    std::pair<cl_platform_id, cl_device_id>& cpu_dev_pair, std::pair<cl_platform_id, cl_device_id>& gpu_dev_pair, timer& time, const size_t gpu_m) {
    cl_int error = CL_SUCCESS;
    cl_context_properties cpu_properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)cpu_dev_pair.first, 0 };
    cl_context_properties gpu_properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)gpu_dev_pair.first, 0 };

    cl_context cpu_context = clCreateContext(cpu_properties, 1, &cpu_dev_pair.second, nullptr, nullptr, &error);
    CONTROL("clCreateContext CPU", error);
    cl_context gpu_context = clCreateContext(gpu_properties, 1, &gpu_dev_pair.second, nullptr, nullptr, &error);
    CONTROL("clCreateContext GPU", error);

    cl_command_queue cpu_queue = clCreateCommandQueue(cpu_context, cpu_dev_pair.second, 0, &error);
    CONTROL("clCreateCommandQueue CPU", error);
    cl_command_queue gpu_queue = clCreateCommandQueue(gpu_context, gpu_dev_pair.second, 0, &error);
    CONTROL("clCreateCommandQueue GPU", error);

    cl_program cpu_program = createProgramFromSource(cpu_context, "kernels/gemm_kernel.cl");
    cl_program gpu_program = createProgramFromSource(gpu_context, "kernels/gemm_kernel.cl");
    std::string build_options = "-DBLOCK=" + std::to_string(BLOCK);
    CONTROL("clBuildProgram CPU", clBuildProgram(cpu_program, 1, &cpu_dev_pair.second, build_options.c_str(), nullptr, nullptr));
    CONTROL("clBuildProgram GPU", clBuildProgram(gpu_program, 1, &gpu_dev_pair.second, build_options.c_str(), nullptr, nullptr));

    cl_kernel cpu_kernel = clCreateKernel(cpu_program, "gemm", &error);
    CONTROL("clCreateKernel CPU", error);
    cl_kernel gpu_kernel = clCreateKernel(gpu_program, "gemm", &error);
    CONTROL("clCreateKernel GPU", error);

    size_t group = BLOCK;
    const size_t cpu_m = m - gpu_m;
    cl_mem gpu_a_buffer = nullptr, gpu_b_buffer = nullptr, gpu_c_buffer = nullptr, cpu_a_buffer = nullptr, cpu_b_buffer = nullptr, cpu_c_buffer = nullptr;
    if (gpu_m > 0) {
        gpu_a_buffer = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizeof(float) * gpu_m * n, nullptr, &error);
        CONTROL("clCreateBuffer A", error);
        gpu_b_buffer = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizeof(float) * n * k, nullptr, &error);
        CONTROL("clCreateBuffer B", error);
        gpu_c_buffer = clCreateBuffer(gpu_context, CL_MEM_WRITE_ONLY, sizeof(float) * gpu_m * k, nullptr, &error);
        CONTROL("clCreateBuffer C", error);
        CONTROL("clEnqueueWriteBuffer A", clEnqueueWriteBuffer(gpu_queue, gpu_a_buffer, CL_TRUE, 0, sizeof(float) * gpu_m * n, a, 0, nullptr, nullptr));
        CONTROL("clEnqueueWriteBuffer B", clEnqueueWriteBuffer(gpu_queue, gpu_b_buffer, CL_TRUE, 0, sizeof(float) * n * k, b, 0, nullptr, nullptr));
        CONTROL("clSetKernelArg M", clSetKernelArg(gpu_kernel, 0, sizeof(unsigned int), &gpu_m));
        CONTROL("clSetKernelArg N", clSetKernelArg(gpu_kernel, 1, sizeof(unsigned int), &n));
        CONTROL("clSetKernelArg K", clSetKernelArg(gpu_kernel, 2, sizeof(unsigned int), &k));
        CONTROL("clSetKernelArg A", clSetKernelArg(gpu_kernel, 3, sizeof(cl_mem), &gpu_a_buffer));
        CONTROL("clSetKernelArg B", clSetKernelArg(gpu_kernel, 4, sizeof(cl_mem), &gpu_b_buffer));
        CONTROL("clSetKernelArg C", clSetKernelArg(gpu_kernel, 5, sizeof(cl_mem), &gpu_c_buffer));
        CONTROL("clGetKernelWorkGroupInf", clGetKernelWorkGroupInfo(gpu_kernel, gpu_dev_pair.second, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr));
    }

    if (cpu_m > 0) {
        cpu_a_buffer = clCreateBuffer(cpu_context, CL_MEM_READ_ONLY, sizeof(float) * cpu_m * n, nullptr, &error);
        CONTROL("clCreateBuffer A", error);
        cpu_b_buffer = clCreateBuffer(cpu_context, CL_MEM_READ_ONLY, sizeof(float) * n * k, nullptr, &error);
        CONTROL("clCreateBuffer B", error);
        cpu_c_buffer = clCreateBuffer(cpu_context, CL_MEM_WRITE_ONLY, sizeof(float) * cpu_m * k, nullptr, &error);
        CONTROL("clCreateBuffer C", error);
        CONTROL("clEnqueueWriteBuffer A", clEnqueueWriteBuffer(cpu_queue, cpu_a_buffer, CL_TRUE, 0, sizeof(float) * cpu_m * n, &a[gpu_m * n], 0, nullptr, nullptr));
        CONTROL("clEnqueueWriteBuffer B", clEnqueueWriteBuffer(cpu_queue, cpu_b_buffer, CL_TRUE, 0, sizeof(float) * n * k, b, 0, nullptr, nullptr));
        CONTROL("clSetKernelArg M", clSetKernelArg(cpu_kernel, 0, sizeof(unsigned int), &cpu_m));
        CONTROL("clSetKernelArg N", clSetKernelArg(cpu_kernel, 1, sizeof(unsigned int), &n));
        CONTROL("clSetKernelArg K", clSetKernelArg(cpu_kernel, 2, sizeof(unsigned int), &k));
        CONTROL("clSetKernelArg A", clSetKernelArg(cpu_kernel, 3, sizeof(cl_mem), &cpu_a_buffer));
        CONTROL("clSetKernelArg B", clSetKernelArg(cpu_kernel, 4, sizeof(cl_mem), &cpu_b_buffer));
        CONTROL("clSetKernelArg C", clSetKernelArg(cpu_kernel, 5, sizeof(cl_mem), &cpu_c_buffer));
        CONTROL("clGetKernelWorkGroupInf", clGetKernelWorkGroupInfo(cpu_kernel, cpu_dev_pair.second, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr));
    }

    const size_t ndims = 2;
    size_t global[ndims] = { k, gpu_m };
    size_t local[ndims] = { BLOCK, BLOCK };

    time.first = std::chrono::high_resolution_clock::now();
    if (gpu_m > 0) {
        CONTROL("clEnqueueNDRangeKernel", clEnqueueNDRangeKernel(gpu_queue, gpu_kernel, ndims, nullptr, global, local, 0, nullptr, nullptr));
    }
   //CONTROL("clFlush GPU", clFlush(gpu_queue));
    if (cpu_m > 0) {
        global[1] = cpu_m;
        CONTROL("clEnqueueNDRangeKernel", clEnqueueNDRangeKernel(cpu_queue, cpu_kernel, ndims, nullptr, global, local, 0, nullptr, nullptr));
    }
    //CONTROL("clFlush CPU", clFlush(cpu_queue));
    CONTROL("clFinish", clFinish(gpu_queue));
    CONTROL("clFinish", clFinish(cpu_queue));
    time.second = std::chrono::high_resolution_clock::now();

    if (gpu_m > 0) {
        clEnqueueReadBuffer(gpu_queue, gpu_c_buffer, CL_TRUE, 0, sizeof(float) * gpu_m * k, c, 0, nullptr, nullptr);
        clReleaseMemObject(gpu_a_buffer);
        clReleaseMemObject(gpu_b_buffer);
        clReleaseMemObject(gpu_c_buffer);
    }
    if (cpu_m > 0) {
        clEnqueueReadBuffer(cpu_queue, cpu_c_buffer, CL_TRUE, 0, sizeof(float) * cpu_m * k, &c[gpu_m * k], 0, nullptr, nullptr);
        clReleaseMemObject(cpu_a_buffer);
        clReleaseMemObject(cpu_b_buffer);
        clReleaseMemObject(cpu_c_buffer);
    }

    clReleaseProgram(cpu_program);    clReleaseProgram(gpu_program);
    clReleaseKernel(cpu_kernel);      clReleaseKernel(gpu_kernel);
    clReleaseCommandQueue(cpu_queue); clReleaseCommandQueue(gpu_queue);
    clReleaseContext(cpu_context);    clReleaseContext(gpu_context);
}

void jacobi_cl(float* a, float* b, float* x0, float* x1, float* norm, int size,
    std::pair<cl_platform_id, cl_device_id>& cpu_dev_pair, std::pair<cl_platform_id, cl_device_id>& gpu_dev_pair, timer& time,
    cl_ulong& kernel_time, const int gpu_m) {
    cl_int error = CL_SUCCESS;
    cl_context_properties cpu_properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)cpu_dev_pair.first, 0 };
    cl_context_properties gpu_properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)gpu_dev_pair.first, 0 };

    cl_context cpu_context = clCreateContext(cpu_properties, 1, &cpu_dev_pair.second, nullptr, nullptr, &error);
    CONTROL("clCreateContext CPU", error);
    cl_context gpu_context = clCreateContext(gpu_properties, 1, &gpu_dev_pair.second, nullptr, nullptr, &error);
    CONTROL("clCreateContext GPU", error);

    cl_queue_properties props[3] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue cpu_queue = clCreateCommandQueueWithProperties(cpu_context, cpu_dev_pair.second, props, &error);
    CONTROL("clCreateCommandQueue CPU", error);
    cl_command_queue gpu_queue = clCreateCommandQueueWithProperties(gpu_context, gpu_dev_pair.second, props, &error);
    CONTROL("clCreateCommandQueue GPU", error);

    cl_program gpu_program = createProgramFromSource(gpu_context, "kernels/jacobi_kernel.cl");
    cl_program cpu_program = createProgramFromSource(cpu_context, "kernels/jacobi_kernel.cl");
    CONTROL("clBuildProgram GPU", clBuildProgram(gpu_program, 1, &gpu_dev_pair.second, nullptr, nullptr, nullptr));
    CONTROL("clBuildProgram CPU", clBuildProgram(cpu_program, 1, &cpu_dev_pair.second, nullptr, nullptr, nullptr));

    cl_kernel gpu_kernel = clCreateKernel(gpu_program, "jacobi", &error);
    CONTROL("clCreateKernel", error);
    cl_kernel cpu_kernel = clCreateKernel(cpu_program, "jacobi", &error);
    CONTROL("clCreateKernel", error);

    const size_t cpu_m = size - gpu_m;
    size_t cpu_global_size = 0, gpu_global_size = 0;
    size_t cpu_group_size = 0, gpu_group_size = 0;
    cl_mem gpu_a_buffer = nullptr, gpu_b_buffer = nullptr, gpu_x0_buffer = nullptr, gpu_x1_buffer = nullptr, gpu_norm_buffer = nullptr;
    cl_mem cpu_a_buffer = nullptr, cpu_b_buffer = nullptr, cpu_x0_buffer = nullptr, cpu_x1_buffer = nullptr, cpu_norm_buffer = nullptr;
    if (gpu_m > 0) {
        const size_t stride = 0;
        gpu_a_buffer = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizeof(float) * size * size, nullptr, &error);
        CONTROL("clCreateBuffer A", error);
        gpu_b_buffer = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizeof(float) * gpu_m, nullptr, &error);
        CONTROL("clCreateBuffer B", error);
        gpu_x0_buffer = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(float) * gpu_m, nullptr, &error);
        CONTROL("clCreateBuffer X0", error);
        gpu_x1_buffer = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(float) * gpu_m, nullptr, &error);
        CONTROL("clCreateBuffer X1", error);
        gpu_norm_buffer = clCreateBuffer(gpu_context, CL_MEM_WRITE_ONLY, sizeof(float) * gpu_m, nullptr, &error);
        CONTROL("clCreateBuffer NORM", error);

        CONTROL("clEnqueueWriteBuffer A", clEnqueueWriteBuffer(gpu_queue, gpu_a_buffer, CL_TRUE, 0, sizeof(float) * size * size, a, 0, nullptr, nullptr));
        CONTROL("clEnqueueWriteBuffer B", clEnqueueWriteBuffer(gpu_queue, gpu_b_buffer, CL_TRUE, 0, sizeof(float) * gpu_m, b, 0, nullptr, nullptr));
        CONTROL("clEnqueueWriteBuffer X0", clEnqueueWriteBuffer(gpu_queue, gpu_x0_buffer, CL_TRUE, 0, sizeof(float) * gpu_m, x0, 0, nullptr, nullptr));
        CONTROL("clEnqueueWriteBuffer X1", clEnqueueWriteBuffer(gpu_queue, gpu_x1_buffer, CL_TRUE, 0, sizeof(float) * gpu_m, x1, 0, nullptr, nullptr));
        CONTROL("clEnqueueWriteBuffer NORM", clEnqueueWriteBuffer(gpu_queue, gpu_norm_buffer, CL_TRUE, 0, sizeof(float) * gpu_m, norm, 0, nullptr, nullptr));

        CONTROL("clSetKernelArg A", clSetKernelArg(gpu_kernel, 0, sizeof(cl_mem), &gpu_a_buffer));
        CONTROL("clSetKernelArg B", clSetKernelArg(gpu_kernel, 1, sizeof(cl_mem), &gpu_b_buffer));
        CONTROL("clSetKernelArg X0", clSetKernelArg(gpu_kernel, 2, sizeof(cl_mem), &gpu_x0_buffer));
        CONTROL("clSetKernelArg X1", clSetKernelArg(gpu_kernel, 3, sizeof(cl_mem), &gpu_x1_buffer));
        CONTROL("clSetKernelArg NORM", clSetKernelArg(gpu_kernel, 4, sizeof(cl_mem), &gpu_norm_buffer));
        CONTROL("clSetKernelArg size", clSetKernelArg(gpu_kernel, 5, sizeof(unsigned int), &gpu_m));
        CONTROL("clSetKernelArg size", clSetKernelArg(gpu_kernel, 6, sizeof(unsigned int), &stride));
        CONTROL("clGetKernelWorkGroupInf", clGetKernelWorkGroupInfo(gpu_kernel, gpu_dev_pair.second, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &gpu_group_size, nullptr));
        gpu_global_size = (gpu_m % gpu_group_size == 0) ? gpu_m : gpu_m + gpu_group_size - gpu_m % gpu_group_size;
    }

    if (cpu_m > 0) {
        cpu_a_buffer = clCreateBuffer(cpu_context, CL_MEM_READ_ONLY, sizeof(float) * size * size, nullptr, &error);
        CONTROL("clCreateBuffer A", error);
        cpu_b_buffer = clCreateBuffer(cpu_context, CL_MEM_READ_ONLY, sizeof(float) * cpu_m, nullptr, &error);
        CONTROL("clCreateBuffer B", error);
        cpu_x0_buffer = clCreateBuffer(cpu_context, CL_MEM_READ_WRITE, sizeof(float) * cpu_m, nullptr, &error);
        CONTROL("clCreateBuffer X0", error);
        cpu_x1_buffer = clCreateBuffer(cpu_context, CL_MEM_READ_WRITE, sizeof(float) * cpu_m, nullptr, &error);
        CONTROL("clCreateBuffer X1", error);
        cpu_norm_buffer = clCreateBuffer(cpu_context, CL_MEM_WRITE_ONLY, sizeof(float) * cpu_m, nullptr, &error);
        CONTROL("clCreateBuffer NORM", error);

        CONTROL("clEnqueueWriteBuffer A", clEnqueueWriteBuffer(cpu_queue, cpu_a_buffer, CL_TRUE, 0, sizeof(float) * size * size, a, 0, nullptr, nullptr));
        CONTROL("clEnqueueWriteBuffer B", clEnqueueWriteBuffer(cpu_queue, cpu_b_buffer, CL_TRUE, 0, sizeof(float) * cpu_m, &b[gpu_m], 0, nullptr, nullptr));
        CONTROL("clEnqueueWriteBuffer X0", clEnqueueWriteBuffer(cpu_queue, cpu_x0_buffer, CL_TRUE, 0, sizeof(float) * cpu_m, &x0[gpu_m], 0, nullptr, nullptr));
        CONTROL("clEnqueueWriteBuffer X1", clEnqueueWriteBuffer(cpu_queue, cpu_x1_buffer, CL_TRUE, 0, sizeof(float) * cpu_m, &x1[gpu_m], 0, nullptr, nullptr));
        CONTROL("clEnqueueWriteBuffer NORM", clEnqueueWriteBuffer(cpu_queue, cpu_norm_buffer, CL_TRUE, 0, sizeof(float) * cpu_m, &norm[gpu_m], 0, nullptr, nullptr));

        CONTROL("clSetKernelArg A", clSetKernelArg(cpu_kernel, 0, sizeof(cl_mem), &cpu_a_buffer));
        CONTROL("clSetKernelArg B", clSetKernelArg(cpu_kernel, 1, sizeof(cl_mem), &cpu_b_buffer));
        CONTROL("clSetKernelArg X0", clSetKernelArg(cpu_kernel, 2, sizeof(cl_mem), &cpu_x0_buffer));
        CONTROL("clSetKernelArg X1", clSetKernelArg(cpu_kernel, 3, sizeof(cl_mem), &cpu_x1_buffer));
        CONTROL("clSetKernelArg NORM", clSetKernelArg(cpu_kernel, 4, sizeof(cl_mem), &cpu_norm_buffer));
        CONTROL("clSetKernelArg size", clSetKernelArg(cpu_kernel, 5, sizeof(unsigned int), &cpu_m));
        CONTROL("clSetKernelArg size", clSetKernelArg(cpu_kernel, 6, sizeof(unsigned int), &gpu_m));
        CONTROL("clGetKernelWorkGroupInf", clGetKernelWorkGroupInfo(cpu_kernel, cpu_dev_pair.second, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &cpu_group_size, nullptr));
        cpu_global_size = (cpu_m % cpu_group_size == 0) ? cpu_m : cpu_m + cpu_group_size - cpu_m % cpu_group_size;
    }

    kernel_time = 0;
    cl_event cpu_evt = nullptr, gpu_evt = nullptr;
    float accuracy = 0.0;
    int iters = 0;

    time.first = std::chrono::high_resolution_clock::now();
    while (true) {
        if (gpu_m > 0) {
            CONTROL("clEnqueueNDRangeKernel", clEnqueueNDRangeKernel(gpu_queue, gpu_kernel, 1, nullptr, &gpu_global_size, &gpu_group_size, 0, nullptr, &gpu_evt));
        }
        if (cpu_m > 0) {
            CONTROL("clEnqueueNDRangeKernel", clEnqueueNDRangeKernel(cpu_queue, cpu_kernel, 1, nullptr, &cpu_global_size, &cpu_group_size, 0, nullptr, &cpu_evt));
        }
        if (gpu_m > 0) {
            CONTROL("clWaitForEvents", clWaitForEvents(1, &gpu_evt));
        }
        if (cpu_m > 0) {
            CONTROL("clWaitForEvents", clWaitForEvents(1, &cpu_evt));
        }

        cl_ulong evt_start_time = 0, evt_end_time = 0;
        if (gpu_m > 0) {
            CONTROL("clGetEventProfilingInfo Start", clGetEventProfilingInfo(gpu_evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &evt_start_time, nullptr));
            CONTROL("clGetEventProfilingInfo End", clGetEventProfilingInfo(gpu_evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &evt_end_time, nullptr));
            kernel_time += evt_end_time - evt_start_time;
        }
        if (cpu_m > 0) {
            CONTROL("clGetEventProfilingInfo Start", clGetEventProfilingInfo(cpu_evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &evt_start_time, nullptr));
            CONTROL("clGetEventProfilingInfo End", clGetEventProfilingInfo(cpu_evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &evt_end_time, nullptr));
            kernel_time += evt_end_time - evt_start_time;
        }

        if (gpu_m > 0) {
            CONTROL("clEnqueueReadBuffer X1", clEnqueueReadBuffer(gpu_queue, gpu_x1_buffer, CL_TRUE, 0, sizeof(float) * gpu_m, x1, 0, NULL, NULL));
            CONTROL("clEnqueueReadBuffer NORM", clEnqueueReadBuffer(gpu_queue, gpu_norm_buffer, CL_TRUE, 0, sizeof(float) * gpu_m, norm, 0, NULL, NULL));
        }
        if (cpu_m > 0) {
            CONTROL("clEnqueueReadBuffer X1", clEnqueueReadBuffer(cpu_queue, cpu_x1_buffer, CL_TRUE, 0, sizeof(float) * cpu_m, &x1[gpu_m], 0, NULL, NULL));
            CONTROL("clEnqueueReadBuffer NORM", clEnqueueReadBuffer(cpu_queue, cpu_norm_buffer, CL_TRUE, 0, sizeof(float) * cpu_m, &norm[gpu_m], 0, NULL, NULL));
        }

        accuracy = std::numeric_limits<float>::min();
        for (size_t i = 0; i < size; ++i) {
            if (fabs(norm[i] / x0[i]) > accuracy)
                accuracy = fabs(norm[i] / x0[i]);
        }
        iters++;

        std::swap(x0, x1);

        if (accuracy < EPS || iters >= MAX_ITERS)
            break;

        if (gpu_m > 0) {
            CONTROL("clEnqueueWriteBuffer X0", clEnqueueWriteBuffer(gpu_queue, gpu_x0_buffer, CL_TRUE, 0, sizeof(float) * gpu_m, x0, 0, NULL, NULL));
        }
        if (cpu_m > 0) {
            CONTROL("clEnqueueWriteBuffer X0", clEnqueueWriteBuffer(cpu_queue, cpu_x0_buffer, CL_TRUE, 0, sizeof(float) * cpu_m, &x0[gpu_m], 0, NULL, NULL));
        }
    }
    time.second = std::chrono::high_resolution_clock::now();

    if (gpu_m > 0) {
        CONTROL("clFinish", clFinish(gpu_queue));
    }
    if (cpu_m > 0) {
        CONTROL("clFinish", clFinish(cpu_queue));
    }

    if (accuracy < EPS)
        std::cout << "[ INFO ] Accuracy (" << accuracy << ") is achieved (iters: " << iters << ")" << std::endl;
    else if (iters >= MAX_ITERS)
        std::cout << "[ INFO ] Accuracy isn't achieved (" << accuracy << "), count of iterations is exceeded" << std::endl;

    if (gpu_m > 0) {
        CONTROL("clEnqueueReadBuffer X1", clEnqueueReadBuffer(gpu_queue, gpu_x1_buffer, CL_TRUE, 0, sizeof(float)* gpu_m, x1, 0, NULL, NULL));
        clReleaseMemObject(gpu_a_buffer);
        clReleaseMemObject(gpu_b_buffer);
        clReleaseMemObject(gpu_x0_buffer);
        clReleaseMemObject(gpu_x1_buffer);
        clReleaseMemObject(gpu_norm_buffer);
    }
    if (cpu_m > 0) {
        CONTROL("clEnqueueReadBuffer X1", clEnqueueReadBuffer(cpu_queue, cpu_x1_buffer, CL_TRUE, 0, sizeof(float)* cpu_m, &x1[gpu_m], 0, NULL, NULL));
        clReleaseMemObject(cpu_a_buffer);
        clReleaseMemObject(cpu_b_buffer);
        clReleaseMemObject(cpu_x0_buffer);
        clReleaseMemObject(cpu_x1_buffer);
        clReleaseMemObject(cpu_norm_buffer);
    }

    clReleaseProgram(cpu_program);    clReleaseProgram(gpu_program);
    clReleaseKernel(cpu_kernel);      clReleaseKernel(gpu_kernel);
    clReleaseCommandQueue(cpu_queue); clReleaseCommandQueue(gpu_queue);
    clReleaseContext(cpu_context);    clReleaseContext(gpu_context);
}
