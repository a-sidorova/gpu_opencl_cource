#include "../include/utils.h"


cl_program createProgramFromSource(cl_context ctx, const char* file) {
    std::fstream kernel_file(file, std::ios::in);
    std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());
    kernel_file.close();
    const char* kernel_code_p = kernel_code.c_str();
    size_t kernel_code_len = kernel_code.size();

    cl_int errorcode = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_code_p, &kernel_code_len, &errorcode);
    CONTROL("clCreateProgramWithSource", errorcode);

    return program;
}

cl_uint getCountAndListOfPlatforms(std::vector<cl_platform_id>& pl) {
    std::cout << "*===================================*" << std::endl
        << "\tPLATFORMS INFO" << std::endl
        << "*===================================*" << std::endl;

    cl_uint platformCount = 0;
    CONTROL("clGetPlatformIDs", clGetPlatformIDs(0, nullptr, &platformCount));

    if (platformCount == 0) {
        THROW_EXCEPTION(std::string("platformCount"), std::string("The count of available platforms is zero"));
    }

    cl_platform_id* platforms = new cl_platform_id[platformCount];
    CONTROL("clGetPlatformIDs", clGetPlatformIDs(platformCount, platforms, nullptr));
    std::cout << "[ INFO ] Platform count: " << platformCount << std::endl;
    for (cl_uint i = 0; i < platformCount; ++i) {
        // Get platform info
        char platformName[128];
        CONTROL("clGetPlatformInfo",
            clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName, nullptr));

        std::cout << "[ INFO ] Platform: " << platformName << std::endl;

        // Get CPU count on platform && info
        cl_uint cpuCount = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, nullptr, &cpuCount);
        cl_device_id* cpus = new cl_device_id[cpuCount];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, cpuCount, cpus, nullptr);
        for (cl_uint j = 0; j < cpuCount; ++j) {
            char cpuName[128];
            clGetDeviceInfo(cpus[j], CL_DEVICE_NAME, 128, cpuName, nullptr);
            std::cout << "[ INFO ]\tCPU: " << cpuName << std::endl;
        }

        // Get GPU count on platform && info
        cl_uint gpuCount = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &gpuCount);
        cl_device_id* gpus = new cl_device_id[gpuCount];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, gpuCount, gpus, nullptr);
        for (cl_uint j = 0; j < gpuCount; ++j) {
            char gpuName[128];
            clGetDeviceInfo(gpus[j], CL_DEVICE_NAME, 128, gpuName, nullptr);
            std::cout << "[ INFO ]\tGPU: " << gpuName << std::endl;
        }
        pl.push_back(platforms[i]);
    }

    delete[] platforms;
    return platformCount;
}

cl_device_id getDevice(cl_device_type type, cl_platform_id& plfrm_id) {
    cl_uint device_count = 0;
    clGetDeviceIDs(plfrm_id, type, 0, nullptr, &device_count);

    if (device_count == 0)
        return nullptr;

    if (device_count > 0) {
        std::vector<cl_device_id> device_vec(device_count);
        clGetDeviceIDs(plfrm_id, type, device_count, device_vec.data(), nullptr);

        if (device_vec.size() > 0) {
            cl_device_id id = device_vec.front();
            return id;
        }
    }
    return nullptr;
}

size_t getTheClosestBiggerDegreeOf2(const size_t x) {
    size_t degree = 1;
    while (true) {
        if (degree >= x) return degree;
        degree *= 2;
    }
    return 0;
}
