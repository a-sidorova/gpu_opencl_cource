#include "../include/axpy.h"


void saxpy(const int& n, const float a, const float* x, const int& incx, float* y, const int& incy) {
    for (int i = 0; i < n; i++)
        y[i * incy] += a * x[i * incx];
}
void daxpy(const int& n, const double a, const double* x, const int& incx, double* y, const int& incy) {
    for (int i = 0; i < n; i++)
        y[i * incy] += a * x[i * incx];
}

void saxpy_omp(const int& n, const float a, const float* x, const int& incx, float* y, const int& incy) {
#pragma omp parallel for num_threads(THREADS)
    for (int i = 0; i < n; i++)
        y[i * incy] += a * x[i * incx];
}

void daxpy_omp(const int& n, const double a, const double* x, const int& incx, double* y, const int& incy) {
#pragma omp parallel for num_threads(THREADS)
    for (int i = 0; i < n; i++)
        y[i * incy] += a * x[i * incx];
}

void saxpy_cl(int n, float a, const float* x, int incx, float* y, int incy, std::pair<cl_platform_id, cl_device_id>& dev_pair,
    std::pair<std::chrono::high_resolution_clock::time_point, std::chrono::high_resolution_clock::time_point>& time) {
    cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)dev_pair.first, 0 };

    cl_context context = clCreateContext(properties, 1, &dev_pair.second, NULL, NULL, NULL);

    cl_command_queue queue = clCreateCommandQueue(context, dev_pair.second, 0, NULL);

    cl_program program = createProgramFromSource(context, "kernels/saxpy_kernel.cl");
    CONTROL("clBuildProgram", clBuildProgram(program, 1, &dev_pair.second, nullptr, nullptr, nullptr));

    cl_kernel kernel = clCreateKernel(program, "saxpy", NULL);

    cl_mem y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * incy * n, NULL, NULL);
    cl_mem x_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * incx * n, NULL, NULL);

    CONTROL("clEnqueueWriteBuffer Y", clEnqueueWriteBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(float) * incy * n, y, 0, NULL, NULL));
    CONTROL("clEnqueueWriteBuffer X", clEnqueueWriteBuffer(queue, x_buffer, CL_TRUE, 0, sizeof(float) * incx * n, x, 0, NULL, NULL));

    CONTROL("clSetKernelArg N", clSetKernelArg(kernel, 0, sizeof(int), &n));
    CONTROL("clSetKernelArg A", clSetKernelArg(kernel, 1, sizeof(float), &a));
    CONTROL("clSetKernelArg X", clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_buffer));
    CONTROL("clSetKernelArg INCX", clSetKernelArg(kernel, 3, sizeof(int), &incx));
    CONTROL("clSetKernelArg Y", clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_buffer));
    CONTROL("clSetKernelArg INCY", clSetKernelArg(kernel, 5, sizeof(int), &incy));

    size_t group = 256;
    size_t size = (n % group == 0) ? n : n + group - n % group;

    time.first = std::chrono::high_resolution_clock::now();
    CONTROL("clEnqueueNDRangeKernel", clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, &group, 0, NULL, NULL));
    CONTROL("clFinisl", clFinish(queue));
    time.second = std::chrono::high_resolution_clock::now();

    clEnqueueReadBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(float) * incy * n, y, 0, NULL, NULL);

    clReleaseMemObject(y_buffer);
    clReleaseMemObject(x_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void daxpy_cl(int n, double a, const double* x, int incx, double* y, int incy, std::pair<cl_platform_id, cl_device_id>& dev_pair,
    std::pair<std::chrono::high_resolution_clock::time_point, std::chrono::high_resolution_clock::time_point>& time) {
    cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)dev_pair.first, 0 };

    cl_context context = clCreateContext(properties, 1, &dev_pair.second, NULL, NULL, NULL);

    cl_command_queue queue = clCreateCommandQueue(context, dev_pair.second, 0, NULL);

    cl_program program = createProgramFromSource(context, "kernels/daxpy_kernel.cl");
    CONTROL("clBuildProgram", clBuildProgram(program, 1, &dev_pair.second, nullptr, nullptr, nullptr));

    cl_kernel kernel = clCreateKernel(program, "daxpy", NULL);

    cl_mem y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * incy * n, NULL, NULL);
    cl_mem x_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * incx * n, NULL, NULL);

    CONTROL("clEnqueueWriteBuffer Y", clEnqueueWriteBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(double) * incy * n, y, 0, NULL, NULL));
    CONTROL("clEnqueueWriteBuffer X", clEnqueueWriteBuffer(queue, x_buffer, CL_TRUE, 0, sizeof(double) * incx * n, x, 0, NULL, NULL));

    CONTROL("clSetKernelArg N", clSetKernelArg(kernel, 0, sizeof(int), &n));
    CONTROL("clSetKernelArg A", clSetKernelArg(kernel, 1, sizeof(double), &a));
    CONTROL("clSetKernelArg X", clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_buffer));
    CONTROL("clSetKernelArg INCX", clSetKernelArg(kernel, 3, sizeof(int), &incx));
    CONTROL("clSetKernelArg Y", clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_buffer));
    CONTROL("clSetKernelArg INCY", clSetKernelArg(kernel, 5, sizeof(int), &incy));

    size_t group = 256;
    size_t size = (n % group == 0) ? n : n + group - n % group;

    time.first = std::chrono::high_resolution_clock::now();
    CONTROL("clEnqueueNDRangeKernel", clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, &group, 0, NULL, NULL));
    CONTROL("clFinisl", clFinish(queue));
    time.second = std::chrono::high_resolution_clock::now();

    clEnqueueReadBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(double) * incy * n, y, 0, NULL, NULL);

    clReleaseMemObject(y_buffer);
    clReleaseMemObject(x_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
