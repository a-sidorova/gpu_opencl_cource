#include <iostream>
#include <chrono>
#include <string>
#include <vector>

#include "exceptions.h"
#include "utils.h"
#include "include/axpy.h"

#define FLAG_CHECK true


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

    const int n = 50'000'000;
    const int inc_x = 1;
    const int inc_y = 1;
    size_t group_size = 256;

    const int x_size = n * inc_x;
    const int y_size = n * inc_y;


    //************************************************************************************
    // FLOAT
    //************************************************************************************
    std::cout << "===========================" << std::endl
        << "\tFLOAT" << std::endl
        << "===========================" << std::endl;
    try {
        float a = 10.0;
        float* x = new float[x_size];
        float* y = new float[y_size];
        float* ref = new float[y_size];
        fillData<float>(x, x_size);

        // SEQ
        fillData<float>(y, y_size);
        auto t0 = std::chrono::high_resolution_clock::now();
        saxpy(n, a, x, inc_x, y, inc_y);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Time 'Sequential': " << TIME_MS(t0, t1) << std::endl;
        std::memcpy(ref, y, y_size * sizeof(float));

        // OMP
        fillData<float>(y, y_size);
        t0 = std::chrono::high_resolution_clock::now();
        saxpy_omp(n, a, x, inc_x, y, inc_y);
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Time 'OMP': " << TIME_MS(t0, t1) << std::endl;
        CHECK(FLAG_CHECK, float, ref, y, y_size)
        
        // GPU OPENCL
        for (int i = 0; i < gpus.size(); i++) {
            fillData<float>(y, y_size);
            timer time;
            saxpy_cl(n, a, x, inc_x, y, inc_y, gpus[i], time);
            char name[128];
            clGetDeviceInfo(gpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
            std::cout << "Time 'GPU' (device " << name << "): " << TIME_MS(time.first, time.second) << std::endl;
            CHECK(FLAG_CHECK, float, ref, y, y_size)
        }

        // CPU OPENCL
        for (int i = 0; i < cpus.size(); i++) {
            fillData<float>(y, y_size);
            timer time;
            saxpy_cl(n, a, x, inc_x, y, inc_y, cpus[i], time);
            char name[128];
            clGetDeviceInfo(cpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
            std::cout << "Time 'CPU' (device " << name << "): " << TIME_MS(time.first, time.second) << std::endl;
            CHECK(FLAG_CHECK, float, ref, y, y_size)
        }
        delete[] x;
        delete[] y;
        delete[] ref;
    }
    catch (Exception& exception) {
        std::cout << exception.what() << std::endl;
    }

    //************************************************************************************
    // DOUBLE
    //************************************************************************************
    std::cout << "===========================" << std::endl
        << "\tDOUBLE" << std::endl
        << "===========================" << std::endl;
    try {
        double a = 10.0;
        double* x = new double[x_size];
        double* y = new double[y_size];
        double* ref = new double[y_size];
        fillData<double>(x, x_size);

        // SEQ
        fillData<double>(y, y_size);
        auto t0 = std::chrono::high_resolution_clock::now();
        daxpy(n, a, x, inc_x, y, inc_y);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Time 'Sequential': " << TIME_MS(t0, t1) << std::endl;
        std::memcpy(ref, y, y_size * sizeof(double));

        // OMP
        fillData<double>(y, y_size);
        t0 = std::chrono::high_resolution_clock::now();
        daxpy_omp(n, a, x, inc_x, y, inc_y);
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Time 'OMP': " << TIME_MS(t0, t1) << std::endl;
        CHECK(FLAG_CHECK, double, ref, y, y_size)
        
        // GPU OPENCL
        for (int i = 0; i < gpus.size(); i++) {
            fillData<double>(y, y_size);
            timer time;
            daxpy_cl(n, a, x, inc_x, y, inc_y, gpus[i], time);
            char name[128];
            clGetDeviceInfo(gpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
            std::cout << "Time 'GPU' (device " << name << "): " << TIME_MS(time.first, time.second) << std::endl;
             CHECK(FLAG_CHECK, double, ref, y, y_size)
        }

        // CPU OPENCL
        for (int i = 0; i < cpus.size(); i++) {
            fillData<double>(y, y_size);
            timer time;
            daxpy_cl(n, a, x, inc_x, y, inc_y, cpus[i], time);
            char name[128];
            clGetDeviceInfo(cpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
            std::cout << "Time 'CPU' (device " << name << "): " << TIME_MS(time.first, time.second) << std::endl;
            CHECK(FLAG_CHECK, double, ref, y, y_size)
        }
        delete[] x;
        delete[] y;
        delete[] ref;
    }
    catch (Exception& exception) {
        std::cout << exception.what() << std::endl;
    }

    return 0;
}
