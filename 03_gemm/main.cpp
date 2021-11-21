#include <iostream>
#include <chrono>
#include <string>
#include <vector>

#include "exceptions.h"
#include "utils.h"
#include "include/matmul.h"

#define FLAG_CHECK true

#define M 720
#define N 720
#define K 720


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

    const size_t a_size = M * N;
    const size_t b_size = N * K;
    const size_t c_size = M * K;
    float* a = new float[a_size];
    float* b = new float[b_size];
    float* c = new float[c_size];
    float* c_ref = new float[c_size];
    fillData<float>(a, a_size);
    fillData<float>(b, b_size);

    //************************************************************************************
    // TASK 1
    //************************************************************************************
    std::cout << "===========================" << std::endl
        << "\tTASK 1 MatMul" << std::endl
        << "===========================" << std::endl;
    try {
        // SEQ
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul(M, N, K, a, b, c);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Time 'Sequential': " << TIME_MS(t0, t1) << std::endl;
        std::memcpy(c_ref, c, c_size * sizeof(float));

        // OMP
        t0 = std::chrono::high_resolution_clock::now();
        matmul_omp(M, N, K, a, b, c);
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Time 'OMP': " << TIME_MS(t0, t1) << std::endl;
        CHECK(FLAG_CHECK, float, c_ref, c, c_size)

        // CL GPU
        for (int i = 0; i < gpus.size(); i++) {
            timer time;
            matmul_cl(M, N, K, a, b, c, gpus[i], time);
            char name[128];
            clGetDeviceInfo(gpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
            std::cout << "Time 'GPU' (device " << name << "): " << TIME_MS(time.first, time.second) << std::endl;
            CHECK(FLAG_CHECK, float, c_ref, c, c_size)
        }

        // CL ÑPU
        for (int i = 0; i < cpus.size(); i++) {
            timer time;
            matmul_cl(M, N, K, a, b, c, cpus[i], time);
            char name[128];
            clGetDeviceInfo(cpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
            std::cout << "Time 'CPU' (device " << name << "): " << TIME_MS(time.first, time.second) << std::endl;
            CHECK(FLAG_CHECK, float, c_ref, c, c_size)
        }
    }
    catch (Exception& exception) {
        std::cout << exception.what() << std::endl;
    }

    //************************************************************************************
    // TASK 2
    //************************************************************************************
    std::cout << "===========================" << std::endl
        << "\tTASK 2 GEMM via buffer" << std::endl
        << "===========================" << std::endl;
    try {
        // CL GPU
        for (int i = 0; i < gpus.size(); i++) {
            timer time;
            gemm_cl(M, N, K, a, b, c, gpus[i], time);
            char name[128];
            clGetDeviceInfo(gpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
            std::cout << "Time 'GPU' (device " << name << "): " << TIME_MS(time.first, time.second) << std::endl;
            CHECK(FLAG_CHECK, float, c_ref, c, c_size)
        }

        // CL ÑPU
        for (int i = 0; i < cpus.size(); i++) {
            timer time;
            gemm_cl(M, N, K, a, b, c, cpus[i], time);
            char name[128];
            clGetDeviceInfo(cpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
            std::cout << "Time 'CPU' (device " << name << "): " << TIME_MS(time.first, time.second) << std::endl;
            CHECK(FLAG_CHECK, float, c_ref, c, c_size)
        }
    }
    catch (Exception& exception) {
        std::cout << exception.what() << std::endl;
    }

    //************************************************************************************
    // TASK 3
    //************************************************************************************
    std::cout << "===========================" << std::endl
        << "\tTASK 3 GEMM via image" << std::endl
        << "===========================" << std::endl;
    try {
        // CL GPU
        for (int i = 0; i < gpus.size(); i++) {
            timer time;
            gemm_cl(M, N, K, a, b, c, gpus[i], time);
            char name[128];
            clGetDeviceInfo(gpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
            std::cout << "Time 'GPU' (device " << name << "): " << TIME_MS(time.first, time.second) << std::endl;
            CHECK(FLAG_CHECK, float, c_ref, c, c_size)
        }

        // CL ÑPU
        for (int i = 0; i < cpus.size(); i++) {
            timer time;
            gemm_cl(M, N, K, a, b, c, cpus[i], time);
            char name[128];
            clGetDeviceInfo(cpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
            std::cout << "Time 'ÑPU' (device " << name << "): " << TIME_MS(time.first, time.second) << std::endl;
            CHECK(FLAG_CHECK, float, c_ref, c, c_size)
        }
    }
    catch (Exception& exception) {
        std::cout << exception.what() << std::endl;
    }

    return 0;
}