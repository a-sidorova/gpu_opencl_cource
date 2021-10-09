cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 14)

include_directories(common)

find_package(OpenCL REQUIRED)

option(LAB01 OFF)
if(LAB01)
    add_subdirectory(lab_01)
endif(LAB01)

option(LAB02 OFF)
if(LAB02)
    add_subdirectory(lab_02)
endif(LAB02)