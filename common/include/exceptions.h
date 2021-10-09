#ifndef _GPU_EXCEPTIONS_H
#define _GPU_EXCEPTIONS_H

#include <iostream>
#include <string>
#include <exception>

#define THROW_EXCEPTION(section, msg) throw Exception(section + ": " + msg);

class Exception : public std::exception {
private:
    std::string msg;
public:
    Exception(std::string _msg) : msg(_msg) {};

    const char* what() const noexcept
    {
        return msg.c_str();
    }
};

void errorProcessing(const std::string& section, const cl_int errcode_ret) {
    switch(errcode_ret) {
        case 0:
            std::cout << "[ SUCCESS ] " << section << std::endl;
            return;
        case CL_DEVICE_NOT_FOUND: {
            THROW_EXCEPTION(section, "CL_DEVICE_NOT_FOUND")
        }
        case CL_DEVICE_NOT_AVAILABLE : {
            THROW_EXCEPTION(section, "CL_DEVICE_NOT_AVAILABLE")
        }
        case CL_COMPILER_NOT_AVAILABLE : {
            THROW_EXCEPTION(section, "CL_COMPILER_NOT_AVAILABLE")
        }
        case CL_MEM_OBJECT_ALLOCATION_FAILURE : {
            THROW_EXCEPTION(section, "CL_MEM_OBJECT_ALLOCATION_FAILURE")
        }
        case CL_OUT_OF_RESOURCES : {
            THROW_EXCEPTION(section, "CL_OUT_OF_RESOURCES")
        }
        case CL_OUT_OF_HOST_MEMORY : {
            THROW_EXCEPTION(section, "CL_OUT_OF_HOST_MEMORY")
        }
        case CL_PROFILING_INFO_NOT_AVAILABLE : {
            THROW_EXCEPTION(section, "CL_PROFILING_INFO_NOT_AVAILABLE")
        }
        case CL_MEM_COPY_OVERLAP : {
            THROW_EXCEPTION(section, "CL_MEM_COPY_OVERLAP")
        }
        case CL_IMAGE_FORMAT_MISMATCH : {
            THROW_EXCEPTION(section, "CL_IMAGE_FORMAT_MISMATCH")
        }
        case CL_IMAGE_FORMAT_NOT_SUPPORTED : {
            THROW_EXCEPTION(section, "CL_IMAGE_FORMAT_NOT_SUPPORTED")
        }
        case CL_BUILD_PROGRAM_FAILURE : {
            THROW_EXCEPTION(section, "CL_BUILD_PROGRAM_FAILURE")
        }
        case CL_MAP_FAILURE : {
            THROW_EXCEPTION(section, "CL_MAP_FAILURE")
        }
        case CL_MISALIGNED_SUB_BUFFER_OFFSET : {
            THROW_EXCEPTION(section, "CL_MISALIGNED_SUB_BUFFER_OFFSET")
        }
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST : {
            THROW_EXCEPTION(section, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST")
        }
        case CL_COMPILE_PROGRAM_FAILURE : {
            THROW_EXCEPTION(section, "CL_COMPILE_PROGRAM_FAILURE")
        }
        case CL_LINKER_NOT_AVAILABLE : {
            THROW_EXCEPTION(section, "CL_LINKER_NOT_AVAILABLE")
        }
        case CL_LINK_PROGRAM_FAILURE : {
            THROW_EXCEPTION(section, "CL_LINK_PROGRAM_FAILURE")
        }
        case CL_DEVICE_PARTITION_FAILED : {
            THROW_EXCEPTION(section, "CL_DEVICE_PARTITION_FAILED")
        }
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE : {
            THROW_EXCEPTION(section, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE")
        }
        case CL_INVALID_VALUE : {
            THROW_EXCEPTION(section, "CL_INVALID_VALUE")
        }
        case CL_INVALID_DEVICE_TYPE : {
            THROW_EXCEPTION(section, "CL_INVALID_DEVICE_TYPE")
        }
        case CL_INVALID_PLATFORM : {
            THROW_EXCEPTION(section, "CL_INVALID_PLATFORM")
        }
        case CL_INVALID_DEVICE : {
            THROW_EXCEPTION(section, "CL_INVALID_DEVICE")
        }
        case CL_INVALID_CONTEXT : {
            THROW_EXCEPTION(section, "CL_INVALID_CONTEXT")
        }
        case CL_INVALID_QUEUE_PROPERTIES : {
            THROW_EXCEPTION(section, "CL_INVALID_QUEUE_PROPERTIES")
        }
        case CL_INVALID_COMMAND_QUEUE : {
            THROW_EXCEPTION(section, "CL_INVALID_COMMAND_QUEUE")
        }
        case CL_INVALID_HOST_PTR : {
            THROW_EXCEPTION(section, "CL_INVALID_HOST_PTR")
        }
        case CL_INVALID_MEM_OBJECT : {
            THROW_EXCEPTION(section, "CL_INVALID_MEM_OBJECT")
        }
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR : {
            THROW_EXCEPTION(section, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR")
        }
        case CL_INVALID_IMAGE_SIZE : {
            THROW_EXCEPTION(section, "CL_INVALID_IMAGE_SIZE")
        }
        case CL_INVALID_SAMPLER : {
            THROW_EXCEPTION(section, "CL_INVALID_SAMPLER")
        }
        case CL_INVALID_BINARY : {
            THROW_EXCEPTION(section, "CL_INVALID_BINARY")
        }
        case CL_INVALID_BUILD_OPTIONS : {
            THROW_EXCEPTION(section, "CL_INVALID_BUILD_OPTIONS")
        }
        case CL_INVALID_PROGRAM : {
            THROW_EXCEPTION(section, "CL_INVALID_PROGRAM")
        }
        case CL_INVALID_PROGRAM_EXECUTABLE : {
            THROW_EXCEPTION(section, "CL_INVALID_PROGRAM_EXECUTABLE")
        }
        case CL_INVALID_KERNEL_DEFINITION : {
            THROW_EXCEPTION(section, "CL_INVALID_KERNEL_DEFINITION")
        }
        case CL_INVALID_KERNEL : {
            THROW_EXCEPTION(section, "CL_INVALID_KERNEL")
        }
        case CL_INVALID_ARG_INDEX : {
            THROW_EXCEPTION(section, "CL_INVALID_ARG_INDEX")
        }
        case CL_INVALID_ARG_VALUE : {
            THROW_EXCEPTION(section, "CL_INVALID_ARG_VALUE")
        }
        case CL_INVALID_ARG_SIZE : {
            THROW_EXCEPTION(section, "CL_INVALID_ARG_SIZE")
        }
        case CL_INVALID_KERNEL_ARGS : {
            THROW_EXCEPTION(section, "CL_INVALID_KERNEL_ARGS")
        }
        case CL_INVALID_WORK_DIMENSION : {
            THROW_EXCEPTION(section, "CL_INVALID_WORK_DIMENSION")
        }
        case CL_INVALID_WORK_GROUP_SIZE : {
            THROW_EXCEPTION(section, "CL_INVALID_WORK_GROUP_SIZE")
        }
        case CL_INVALID_WORK_ITEM_SIZE : {
            THROW_EXCEPTION(section, "CL_INVALID_WORK_ITEM_SIZE")
        }
        case CL_INVALID_GLOBAL_OFFSET : {
            THROW_EXCEPTION(section, "CL_INVALID_GLOBAL_OFFSET")
        }
        case CL_INVALID_EVENT_WAIT_LIST : {
            THROW_EXCEPTION(section, "CL_INVALID_EVENT_WAIT_LIST")
        }
        case CL_INVALID_EVENT : {
            THROW_EXCEPTION(section, "CL_INVALID_EVENT")
        }
        case CL_INVALID_OPERATION : {
            THROW_EXCEPTION(section, "CL_INVALID_OPERATION")
        }
        case CL_INVALID_GL_OBJECT : {
            THROW_EXCEPTION(section, "CL_INVALID_GL_OBJECT")
        }
        case CL_INVALID_BUFFER_SIZE : {
            THROW_EXCEPTION(section, "CL_INVALID_BUFFER_SIZE")
        }
        case CL_INVALID_MIP_LEVEL : {
            THROW_EXCEPTION(section, "CL_INVALID_MIP_LEVEL")
        }
        case CL_INVALID_GLOBAL_WORK_SIZE : {
            THROW_EXCEPTION(section, "CL_INVALID_GLOBAL_WORK_SIZE")
        }
        case CL_INVALID_PROPERTY : {
            THROW_EXCEPTION(section, "CL_INVALID_PROPERTY")
        }
        case CL_INVALID_IMAGE_DESCRIPTOR : {
            THROW_EXCEPTION(section, "CL_INVALID_IMAGE_DESCRIPTOR")
        }
        case CL_INVALID_COMPILER_OPTIONS : {
            THROW_EXCEPTION(section, "CL_INVALID_COMPILER_OPTIONS")
        }
        case CL_INVALID_LINKER_OPTIONS : {
            THROW_EXCEPTION(section, "CL_INVALID_LINKER_OPTIONS")
        }
        case CL_INVALID_DEVICE_PARTITION_COUNT : {
            THROW_EXCEPTION(section, "CL_INVALID_DEVICE_PARTITION_COUNT")
        }
        case CL_INVALID_KERNEL_NAME : {
            THROW_EXCEPTION(section, "CL_INVALID_KERNEL_NAME")
        }
        default : {
            THROW_EXCEPTION(section, std::to_string(errcode_ret))
        }
    }
}

#endif //_GPU_EXCEPTIONS_H
