project(lab_01)

add_executable(lab_01 main.cpp ../lab_02/main.cpp)
target_link_libraries(lab_01 OpenCL::OpenCL)