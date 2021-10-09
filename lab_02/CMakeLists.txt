project(lab_02)

include_directories(include)

add_executable(lab_02 main.cpp)
target_link_libraries(lab_02 OpenCL::OpenCL)