#include <CL/cl.h>
#include <iostream>

#include "../common/include/utils.h"

#define SIZE 1024

const char* kernelSource =
"__kernel void addThreadGlobalIndex(__global float *input, __global float *output, const unsigned int size) {"\
"    int globalID = get_global_id(0);                                                                        "\
"    int localID  = get_local_id(0);                                                                         "\
"    int groupID  = get_group_id(0);                                                                         "\
"                                                                                                            "\
"    printf(\"I am from %d block, %d thread (global index: %d)\\n\", groupID, localID, globalID);               "\
"                                                                                                            "\
"    if (globalID < size)                                                                                    "\
"        output[globalID] = input[globalID] + globalID;                                                      "\
"}";

int main() {
    try {
        cl_uint platformCount = 0;
        clGetPlatformIDs(0, nullptr, &platformCount);

        if (platformCount == 0) {
            THROW_EXCEPTION(std::string("platformCount"), std::string("The count of available platforms is zero"));
        }

        cl_platform_id *platforms = new cl_platform_id[platformCount];
        CONTROL("clGetPlatformIDs", clGetPlatformIDs(platformCount, platforms, nullptr));

        for (cl_uint i = 0; i < platformCount; ++i) {
            char platformName[128];
            CONTROL("clGetPlatformInfo",
                             clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName, nullptr));

            std::cout << "[ INFO ] Platform: " << platformName << std::endl;
        }

        cl_platform_id platform = platforms[1];
        delete[] platforms;

        cl_context_properties properties[3] = {
                CL_CONTEXT_PLATFORM,
                (cl_context_properties) platform,
                0
        };

        cl_int errorcode = 0;
        size_t size = 0;
        cl_context context = clCreateContextFromType((platform == nullptr) ? nullptr : properties, CL_DEVICE_TYPE_GPU,
                                                     nullptr, nullptr, &errorcode);
        CONTROL("clCreateContextFromType", errorcode);

        CONTROL("clGetContextInfo",
                        clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &size));

        cl_device_id device = nullptr;
        if (size > 0) {
            cl_device_id *devices = (cl_device_id *) alloca(size);
            CONTROL("clGetDeviceContextInfo", clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, 0));

            device = devices[0];
        }

        cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &errorcode);
        CONTROL("clCreateCommandQueueWithProperties", errorcode);

        size_t srclen[] = { strlen(kernelSource) };
        cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, srclen, &errorcode);
        CONTROL("clCreateProgramWithSource", errorcode);

        CONTROL("clBuildProgram",
                        clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr));

        cl_kernel kernel = clCreateKernel(program, "addThreadGlobalIndex", &errorcode);
        CONTROL("clCreateKernel", errorcode);

        float data[SIZE] = {0};
        float results[SIZE] = {0};
        fillData(data, SIZE);
        std::cout << "[ INFO ] results[0] before: " << results[0] << std::endl;

        cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * SIZE, nullptr, &errorcode);
        CONTROL("clCreateInputBuffer", errorcode);
        cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * SIZE, nullptr, &errorcode);
        CONTROL("clCreateOutputBuffer", errorcode);

        CONTROL("clEnqueueWriteBuffer",
                        clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, sizeof(float) * SIZE, data, 0, nullptr, nullptr));

        size_t count = SIZE;
        CONTROL("clSetFirstKernelArg",
                        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input));
        CONTROL("clSetSecondKernelArg",
                        clSetKernelArg(kernel, 1, sizeof(cl_mem), &output));
        CONTROL("clSetThirdKernelArg",
                        clSetKernelArg(kernel, 2, sizeof(size_t), &count));

        size_t group;
        CONTROL("clGetKernelWorkGroupInfo",
                        clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr));
        CONTROL("clEnqueueNDRangeKernel",
                        clEnqueueNDRangeKernel(queue, kernel, 1, 0, &count, &group, 0, nullptr, nullptr));
        CONTROL("clFinish",
                        clFinish(queue));

        CONTROL("clEnqueueReadBuffer",
                        clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * count, results, 0, nullptr, nullptr));

        CONTROL("clReleaseMemInputObject", clReleaseMemObject(input));
        CONTROL("clReleaseMemOutputObject", clReleaseMemObject(output));
        CONTROL("clReleaseProgram", clReleaseProgram(program));
        CONTROL("clReleaseKernel", clReleaseKernel(kernel));
        CONTROL("clReleaseCommandQueue", clReleaseCommandQueue(queue));
        CONTROL("clReleaseContext", clReleaseContext(context));

        std::cout << "[ INFO ] data before: " << data[0] << std::endl;
        std::cout << "[ INFO ] results after: " << results[0] << std::endl;

        return 0;
    }
    catch (const Exception& ex)
    {
        std::cout << "[ ERROR ] " << ex.what() << std::endl;
    }
}