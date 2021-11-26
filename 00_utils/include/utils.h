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

using us = std::chrono::microseconds;
using ms = std::chrono::milliseconds;
using s = std::chrono::seconds;
using timer = std::pair<std::chrono::high_resolution_clock::time_point, std::chrono::high_resolution_clock::time_point>;

cl_program createProgramFromSource(cl_context ctx, const char* file);
cl_uint getCountAndListOfPlatforms(std::vector<cl_platform_id>& pl);
cl_device_id getDevice(cl_device_type type, cl_platform_id& plfrm_id);

size_t getTheClosestBiggerDegreeOf2(const size_t x);


// ------------------------------------------------------------------------------------
// Functions for generate data
template <typename T>
void fillData(T* data, const size_t size) {
    for (int i = 0; i < size; ++i)
        data[i] = .1f * (i % 10) / 128;
}

template <typename T>
void generateVector(T* data, const size_t size) {
    static std::random_device rd;
    static std::default_random_engine re(rd());
    static std::uniform_real_distribution<T> dist{ 2.f, 3.f };
    for (int i = 0; i < size; ++i)
        data[i] = dist(rd);
}

template <typename T>
void generateSymmetricPositiveMatrix(T* matrix, int size) {
    static std::random_device rd;
    static std::default_random_engine re(rd());
    static std::uniform_real_distribution<T> dist{ 2.f, 3.f };
    static std::uniform_real_distribution<T> dist_diag{ size * 3.f, size * 3.f + 1.f };

    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            float tmp = dist(rd);
            if (i != j)
                matrix[i * size + j] = tmp;
        }
        matrix[i * size + i] = dist_diag(rd);
    }
}

// ------------------------------------------------------------------------------------
// Functions for check for accuracy
template <typename T>
bool checkCorrect(T* actual, T* reference, int size) {
    std::cout << std::fixed;
    std::cout.precision(6);
    for (int i = 0; i < size; ++i)
        if (std::abs(actual[i] - reference[i]) >= std::numeric_limits<T>::epsilon()) {
            std::cout << "index: " << i << " expected: " << reference[i] << " vs actual: " << actual[i] << std::endl;
            return false;
        }
    return true;
}

template <typename T>
void checkSolutionOfSOLE(const size_t size, const T* A, const T* b, const T* x, const float eps) {
    std::cout << std::fixed;
    std::cout.precision(6);

    std::vector<T> actual(size, 0);
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            actual[i] += A[j * size + i] * x[j];
        }
    }

    float accuracy = std::numeric_limits<float>::min();
    for (size_t i = 0; i < size; ++i) {
        if (fabs(actual[i] - b[i]) > accuracy)
            accuracy = fabs(actual[i] - b[i]);
    }
    const bool result = accuracy <= eps;
    std::cout << "[ CHECK ]  EPS - Achieved: " << accuracy << "\t" << "EPS - Expected: " << eps << std::endl;
    std::cout << "-- Check-status: " << result << std::endl;
}

// ------------------------------------------------------------------------------------
// Print the system: matrices and vectors.
template <typename T>
void printSystem(T* A, T* b, int size) {
    if (size <= 16) {
        std::cout << "System of equations:" << std::endl;
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                std::cout << A[i * size + j] << "  ";
            }
            std::cout << "|  " << b[i] << std::endl;
        }
        std::cout << std::endl;
    }
}

template <typename T>
void printVector(T* vector, int size) {
    if (size <= 16) {
        std::cout << "Vector:" << std::endl;
        for (size_t i = 0; i < size; i++) {
            std::cout << vector[i] << "\t";
        }
        std::cout << std::endl;
    }
}

#endif //_GPU_UTILS_H

