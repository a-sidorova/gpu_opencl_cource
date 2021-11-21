#include "../include/jacobi.h"

void jacobi_cl(float* a, float* b, float* x0, float* x1, float* norm, int size,
    cl_device_type device_type, std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time, cl_ulong& kernel_time) {
    cl_int error = CL_SUCCESS;
    cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)dev_pair.first, 0 };

    cl_context context = clCreateContextFromType((nullptr == dev_pair.first) ? nullptr : properties,
        device_type, nullptr, nullptr, &error);
    CONTROL("clCreateContext", error);

    cl_queue_properties props[3] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, dev_pair.second, props, &error);
    CONTROL("clCreateCommandQueue", error);

    cl_program program = createProgramFromSource(context, "kernels/jacobi_kernel.cl");
    CONTROL("clBuildProgram", clBuildProgram(program, 1, &dev_pair.second, nullptr, nullptr, nullptr));

    cl_kernel kernel = clCreateKernel(program, "jacobi", &error);
    CONTROL("clCreateKernel", error);

    cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * size * size, nullptr, &error);
    CONTROL("clCreateBuffer A", error);
    cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * size, nullptr, &error);
    CONTROL("clCreateBuffer B", error);
    cl_mem x0_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size, nullptr, &error);
    CONTROL("clCreateBuffer X0", error);
    cl_mem x1_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size, nullptr, &error);
    CONTROL("clCreateBuffer X1", error);
    cl_mem norm_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, nullptr, &error);
    CONTROL("clCreateBuffer NORM", error);

    CONTROL("clEnqueueWriteBuffer A", clEnqueueWriteBuffer(queue, a_buffer, CL_TRUE, 0, sizeof(float) * size * size, a, 0, nullptr, nullptr));
    CONTROL("clEnqueueWriteBuffer B", clEnqueueWriteBuffer(queue, b_buffer, CL_TRUE, 0, sizeof(float) * size, b, 0, nullptr, nullptr));
    CONTROL("clEnqueueWriteBuffer X0", clEnqueueWriteBuffer(queue, x0_buffer, CL_TRUE, 0, sizeof(float) * size, x0, 0, nullptr, nullptr));
    CONTROL("clEnqueueWriteBuffer X1", clEnqueueWriteBuffer(queue, x1_buffer, CL_TRUE, 0, sizeof(float) * size, x1, 0, nullptr, nullptr));
    CONTROL("clEnqueueWriteBuffer NORM", clEnqueueWriteBuffer(queue, norm_buffer, CL_TRUE, 0, sizeof(float) * size, norm, 0, nullptr, nullptr));

    CONTROL("clSetKernelArg A", clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buffer));
    CONTROL("clSetKernelArg B", clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buffer));
    CONTROL("clSetKernelArg X0", clSetKernelArg(kernel, 2, sizeof(cl_mem), &x0_buffer));
    CONTROL("clSetKernelArg X1", clSetKernelArg(kernel, 3, sizeof(cl_mem), &x1_buffer));
    CONTROL("clSetKernelArg NORM", clSetKernelArg(kernel, 4, sizeof(cl_mem), &norm_buffer));
    CONTROL("clSetKernelArg size", clSetKernelArg(kernel, 5, sizeof(unsigned int), &size));

    size_t group_size = 0;
    clGetKernelWorkGroupInfo(kernel, dev_pair.second, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group_size, nullptr);
    const size_t global_size = (size % group_size == 0) ? size : size + group_size - size % group_size;

    kernel_time = 0;
    cl_event evt;
    float accuracy = 0.0;
    int iters = 0;

    time.first = std::chrono::high_resolution_clock::now();
    while (true) {
        CONTROL("clEnqueueNDRangeKernel", clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &group_size, 0, nullptr, &evt));
        CONTROL("clWaitForEvents", clWaitForEvents(1, &evt));

        cl_ulong evt_start_time = 0, evt_end_time = 0;
        CONTROL("clGetEventProfilingInfo Start", clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &evt_start_time, nullptr));
        CONTROL("clGetEventProfilingInfo End", clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &evt_end_time, nullptr));
        kernel_time += evt_end_time - evt_start_time;

        CONTROL("clEnqueueReadBuffer X1", clEnqueueReadBuffer(queue, x1_buffer, CL_TRUE, 0, sizeof(float) * size, x1, 0, NULL, NULL));
        CONTROL("clEnqueueReadBuffer NORM", clEnqueueReadBuffer(queue, norm_buffer, CL_TRUE, 0, sizeof(float) * size, norm, 0, NULL, NULL));

        accuracy = std::numeric_limits<float>::min();
        for (size_t i = 0; i < size; ++i) {
            if ((fabs(norm[i]) / fabs(x1[i])) > accuracy)
                accuracy = (fabs(norm[i]) / fabs(x1[i]));
        }
        iters++;

        std::swap(x0, x1);

        if (accuracy < EPS || iters >= MAX_ITERS)
            break;

        CONTROL("clEnqueueWriteBuffer X0", clEnqueueWriteBuffer(queue, x0_buffer, CL_TRUE, 0, sizeof(float) * size, x0, 0, NULL, NULL));
    }
    time.second = std::chrono::high_resolution_clock::now();

    CONTROL("clFinish", clFinish(queue));


    CONTROL("clEnqueueReadBuffer", clEnqueueReadBuffer(queue, x0_buffer, CL_TRUE, 0, size * sizeof(float), x1, 0, nullptr, nullptr));

    if (accuracy < EPS)
        std::cout << "[ INFO ] Accuracy (" << accuracy << ") is achieved (iters: " << iters << ")" << std::endl;
    else if (iters > MAX_ITERS)
        std::cout << "[ INFO ] Accuracy isn't achieved (" << accuracy << "), count of iterations is exceeded" << std::endl;

    clReleaseMemObject(a_buffer);
    clReleaseMemObject(b_buffer);
    clReleaseMemObject(x0_buffer);
    clReleaseMemObject(x1_buffer);
    clReleaseMemObject(norm_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}