#include <iostream>
#include <chrono>
#include <string>
#include <vector>

#include "exceptions.h"
#include "utils.h"
#include "include/hetero_algorithms.h"

#define FLAG_CHECK true
#define SIZE 2048

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

    //************************************************************************************
    // TASK 1
    //************************************************************************************
    {
        std::cout << "===========================" << std::endl
            << "\tTASK 1 GEMM" << std::endl
            << "===========================" << std::endl;
        const size_t a_size = M * N;
        const size_t b_size = N * K;
        const size_t c_size = M * K;
        float* a = new float[a_size];
        float* b = new float[b_size];
        float* c = new float[c_size];
        float* c_ref = new float[c_size];
        fillData<float>(a, a_size);
        fillData<float>(b, b_size);

        try {
            // SEQ
            {
                matmul(M, N, K, a, b, c);
                std::memcpy(c_ref, c, c_size * sizeof(float));
            }

            // CPU
            {
                timer time;
                const size_t gpu_m = 0;
                gemm_cl(M, N, K, a, b, c, cpus[0], gpus[0], time, gpu_m);
                char cpu_name[128];
                clGetDeviceInfo(cpus[0].second, CL_DEVICE_NAME, 128, cpu_name, nullptr);
                std::cout << "Time 'CPU' (device " << cpu_name << "): " << TIME_MS(time.first, time.second) << std::endl;
                CHECK(FLAG_CHECK, float, c_ref, c, c_size);
            }

            // GPU
            {
                timer time;
                const size_t gpu_m = M;
                gemm_cl(M, N, K, a, b, c, gpus[0], gpus[0], time, gpu_m);
                char gpu_name[128];
                clGetDeviceInfo(gpus[0].second, CL_DEVICE_NAME, 128, gpu_name, nullptr);
                std::cout << "Time 'CPU' (device " << gpu_name << "): " << TIME_MS(time.first, time.second) << std::endl;
                CHECK(FLAG_CHECK, float, c_ref, c, c_size);
            }

            // CPU and GPU 0.8
            {
                timer time;
                const size_t gpu_m = M * 0.8;
                gemm_cl(M, N, K, a, b, c, cpus[0], gpus[0], time, gpu_m);
                char cpu_name[128];
                char gpu_name[128];
                clGetDeviceInfo(cpus[0].second, CL_DEVICE_NAME, 128, cpu_name, nullptr);
                clGetDeviceInfo(gpus[0].second, CL_DEVICE_NAME, 128, gpu_name, nullptr);
                std::cout << "Time 'GPU' (device " << gpu_name << ") and 'CPU' (device " << cpu_name << "), GPU = 0.8%: " << TIME_MS(time.first, time.second) << std::endl;
                CHECK(FLAG_CHECK, float, c_ref, c, c_size);
            }

            // CPU and GPU 0.6
            {
                timer time;
                const size_t gpu_m = M * 0.6;
                gemm_cl(M, N, K, a, b, c, cpus[0], gpus[0], time, gpu_m);
                char cpu_name[128];
                char gpu_name[128];
                clGetDeviceInfo(cpus[0].second, CL_DEVICE_NAME, 128, cpu_name, nullptr);
                clGetDeviceInfo(gpus[0].second, CL_DEVICE_NAME, 128, gpu_name, nullptr);
                std::cout << "Time 'GPU' (device " << gpu_name << ") and 'CPU' (device " << cpu_name << "), GPU = 0.6%: " << TIME_MS(time.first, time.second) << std::endl;
                CHECK(FLAG_CHECK, float, c_ref, c, c_size);
            }

            // CPU and GPU 0.5
            {
                timer time;
                const size_t gpu_m = 368;
                gemm_cl(M, N, K, a, b, c, cpus[0], gpus[0], time, gpu_m);
                char cpu_name[128];
                char gpu_name[128];
                clGetDeviceInfo(cpus[0].second, CL_DEVICE_NAME, 128, cpu_name, nullptr);
                clGetDeviceInfo(gpus[0].second, CL_DEVICE_NAME, 128, gpu_name, nullptr);
                std::cout << "Time 'GPU' (device " << gpu_name << ") and 'CPU' (device " << cpu_name << "), GPU = 0.51%: " << TIME_MS(time.first, time.second) << std::endl;
                CHECK(FLAG_CHECK, float, c_ref, c, c_size);
            }
        }
        catch (Exception& exc) {
            std::cout << exc.what() << std::endl;
        }
    }

    return 0;
}