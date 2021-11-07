#ifndef _GPU_UTILS_H
#define _GPU_UTILS_H

#include <CL/cl.h>
#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <chrono>

#include "exceptions.h"

#define CONTROL(section, errcode) errorProcessing(section, errcode)
#define CHECK(flag, type, reference, result, size)                 \
    if (flag) {                                              \
    bool res = checkCorrect<type>(result, reference, size); \
    std::cout << "-- Check-status: " << res << std::endl;    \
}
#define TIME_US(t0, t1) std::chrono::duration_cast<us>(t1 - t0).count() << " us"
#define TIME_MS(t0, t1) std::chrono::duration_cast<ms>(t1 - t0).count() << " ms"
#define TIME_S(t0, t1) std::chrono::duration_cast<s>(t1 - t0).count() << " s"

typedef std::chrono::microseconds us;
typedef std::chrono::milliseconds ms;
typedef std::chrono::seconds s;

cl_program createProgramFromSource(cl_context ctx, const char* file);
cl_uint getCountAndListOfPlatforms(std::vector<cl_platform_id>& pl);
cl_device_id getDevice(cl_device_type type, cl_platform_id& plfrm_id);

template <typename T>
void fillData(T* data, const size_t size) {
    for (int i = 0; i < size; ++i)
        data[i] = .1f * (i % 10) + 5.f;
}

template <typename T>
bool checkCorrect(T* actual, T* reference, int size) {
    T eps = 1e-3;
    bool status = true;
    for (int i = 0; status && i < size; ++i)
        status = status && (std::abs(actual[i] - reference[i]) < eps);
    return status;
}

#endif //_GPU_UTILS_H

