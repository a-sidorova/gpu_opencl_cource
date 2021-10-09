#ifndef _GPU_UTILS_H
#define _GPU_UTILS_H

#include <CL/cl.h>
#include <iostream>
#include <random>

#include "exceptions.h"

#define CONTROL(section, errcode) errorProcessing(section, errcode);

void fillData(float* data, const size_t size) {
    std::mt19937 gen;
    gen.seed(static_cast<size_t>(time(0)));
    for (int i = 0; i < size; ++i)
        data[i] = rand();
}


#endif //_GPU_UTILS_H
