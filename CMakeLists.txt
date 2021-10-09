cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 14)

include_directories(common)

find_package(OpenCL REQUIRED)

option(LAB01 OFF)
if(LAB01)
    add_subdirectory(lab_01)
endif(LAB01)
