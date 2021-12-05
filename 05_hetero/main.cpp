#include <iostream>
#include <chrono>
#include <string>
#include <vector>

#include "exceptions.h"
#include "utils.h"
#include "include/hetero_algorithms.h"

#define FLAG_CHECK true
#define SIZE 4096

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

    const std::vector<float> percents = { 0, 1, 0.25, 0.5, 0.6, 0.75, 0.8, 0.9 };

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

        auto getTheClosestNumber = [](const size_t number, const size_t div) -> size_t {
            size_t new_number = number;
            while (new_number % div != 0)
                new_number++;
            return new_number;
        };

        try {
            // SEQ
            {
                matmul(M, N, K, a, b, c);
                std::memcpy(c_ref, c, c_size * sizeof(float));
            }

            for (auto pers : percents) {
                timer time;
                const size_t gpu_m = getTheClosestNumber(M * pers, BLOCK);
                char cpu_name[128];
                char gpu_name[128];
                clGetDeviceInfo(cpus[0].second, CL_DEVICE_NAME, 128, cpu_name, nullptr);
                clGetDeviceInfo(gpus[0].second, CL_DEVICE_NAME, 128, gpu_name, nullptr);
                std::cout << "Time 'GPU' (device " << gpu_name << ") and 'CPU' (device " << cpu_name << "), GPU = " << pers << "%: ";
                gemm_cl(M, N, K, a, b, c, cpus[0], gpus[0], time, gpu_m);
                std::cout << TIME_MS(time.first, time.second) << std::endl;
                CHECK(FLAG_CHECK, float, c_ref, c, c_size);
            }
        }
        catch (Exception& exc) {
            std::cout << exc.what() << std::endl;
        }
    }


    //************************************************************************************
    // TASK 2
    //************************************************************************************
    {
        std::cout << "===========================" << std::endl
            << "\tTASK 2 JACOBI" << std::endl
            << "===========================" << std::endl;
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

        try {
            for (auto pers : percents) {
                std::memcpy(x0, tmp, SIZE * sizeof(float));
                std::memset(norm, 0, sizeof(float) * SIZE);
                std::memset(x1, 0, sizeof(float) * SIZE);

                timer time;
                const size_t gpu_m = SIZE * pers;
                char cpu_name[128];
                char gpu_name[128];
                clGetDeviceInfo(cpus[0].second, CL_DEVICE_NAME, 128, cpu_name, nullptr);
                clGetDeviceInfo(gpus[0].second, CL_DEVICE_NAME, 128, gpu_name, nullptr);
                std::cout << "Time 'GPU' (device " << gpu_name << ") and 'CPU' (device " << cpu_name << "), GPU = " << pers << "%:\n";
                jacobi_cl(a, b, x0, x1, norm, SIZE, cpus[0], gpus[0], time, kernel_time, gpu_m);
                std::cout << "-- all actions: " << TIME_MS(time.first, time.second) << "\n" <<
                    "-- only kernel: " << kernel_time * 1e-06 << " ms" << std::endl;
                if (FLAG_CHECK)
                    checkSolutionOfSOLE(SIZE, a, b, x1, EPS);
                std::cout << std::endl;
            }
        }
        catch (Exception& exc) {
            std::cout << exc.what() << std::endl;
        }
    }

    return 0;
}
