#include <iostream>
#include <chrono>
#include <string>
#include <vector>

#include "include/jacobi.h"
#include "exceptions.h"
#include "utils.h"

#define FLAG_CHECK true
#define SIZE 2048

int main(int argc, char** argv) {
    std::vector<cl_platform_id> platforms;
    cl_uint platformCount = getCountAndListOfPlatforms(platforms);
    std::vector<std::pair<cl_platform_id, cl_device_id>> gpus, cpus;
    if (platformCount < 1) {
        THROW_EXCEPTION(std::string("PlatfromCount"), std::to_string(platformCount))
    }

    for (size_t i = 0; i < platformCount; i++) {
        cl_platform_id platform = platforms[i];

        cl_device_id gpu = getDevice(CL_DEVICE_TYPE_GPU, platform);
        if (gpu != nullptr)
            gpus.push_back(std::make_pair(platform, gpu));

        cl_device_id cpu = getDevice(CL_DEVICE_TYPE_CPU, platform);
        if (cpu != nullptr)
            cpus.push_back(std::make_pair(platform, cpu));
    }

    // SRC DATA
    float* a = new float[SIZE * SIZE];
    float* b = new float[SIZE];
    float* x0 = new float[SIZE];
    float* x1 = new float[SIZE];
    float* norm = new float[SIZE];
    float* tmp = new float[SIZE];
    cl_ulong kernel_time = 0;
    timer time;

    // System of equations
    generateSymmetricPositiveMatrix(a, SIZE);
    generateVector(b, SIZE);
    printSystem(a, b, SIZE);
    generateVector(tmp, SIZE);

    // GPU OPENCL
    try {
        std::cout << "===========================" << std::endl
            << "\tGPU OPENCL" << std::endl
            << "===========================" << std::endl;
        for (size_t i = 0; i < gpus.size(); i++) {
            std::memcpy(x0, tmp, SIZE * sizeof(float));
            std::memset(norm, 0, sizeof(float) * SIZE);
            std::memset(x1, 0, sizeof(float) * SIZE);

            char name[128];
            clGetDeviceInfo(gpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
            std::cout << "Time 'GPU' (device " << name << ")" << std::endl;

            jacobi_cl(a, b, x0, x1, norm, SIZE, CL_DEVICE_TYPE_GPU, gpus[i], time, kernel_time);

            std::cout << "-- all actions: " << TIME_MS(time.first, time.second) << "\n" <<
                "-- only kernel: " << kernel_time * 1e-06 << " ms" << std::endl;
            if (FLAG_CHECK)
                checkSolutionOfSOLE(SIZE, a, b, x1, EPS);
            std::cout << std::endl;
        }
    } catch (Exception& exception) {
        std::cout << exception.what() << std::endl;
    }

    // CPU OPENCL
    try {
        std::cout << "===========================" << std::endl
            << "\tCPU OPENCL" << std::endl
            << "===========================" << std::endl;
        for (size_t i = 0; i < cpus.size(); i++) {
            std::memcpy(x0, tmp, SIZE * sizeof(float));
            std::memset(norm, 0, sizeof(float) * SIZE);
            std::memset(x1, 0, sizeof(float) * SIZE);

            char name[128];
            clGetDeviceInfo(cpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
            std::cout << "Time 'CPU' (device " << name << ")" << std::endl;

            jacobi_cl(a, b, x0, x1, norm, SIZE, CL_DEVICE_TYPE_CPU, cpus[i], time, kernel_time);

            std::cout << "-- all actions: " << TIME_MS(time.first, time.second) << "\n" <<
                "-- only kernel: " << kernel_time * 1e-06 << " ms" << std::endl;
            if (FLAG_CHECK)
                checkSolutionOfSOLE(SIZE, a, b, x1, EPS);
            std::cout << std::endl;
        }
    } catch (Exception& exception) {
        std::cout << exception.what() << std::endl;
    }

    delete[] a;
    delete[] b;
    delete[] x0;
    delete[] x1;
    delete[] norm;
    delete[] tmp;

    return 0;
}