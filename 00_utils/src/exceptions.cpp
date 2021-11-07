#include "../include/exceptions.h"

extern void errorProcessing(const std::string& section, const cl_int errcode_ret) {
    switch (errcode_ret) {
    case 0:
        return;
    case CL_DEVICE_NOT_FOUND: {
        THROW_EXCEPTION(section, "CL_DEVICE_NOT_FOUND")
    }
    case CL_DEVICE_NOT_AVAILABLE: {
        THROW_EXCEPTION(section, "CL_DEVICE_NOT_AVAILABLE")
    }
    case CL_COMPILER_NOT_AVAILABLE: {
        THROW_EXCEPTION(section, "CL_COMPILER_NOT_AVAILABLE")
    }
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: {
        THROW_EXCEPTION(section, "CL_MEM_OBJECT_ALLOCATION_FAILURE")
    }
    case CL_OUT_OF_RESOURCES: {
        THROW_EXCEPTION(section, "CL_OUT_OF_RESOURCES")
    }
    case CL_OUT_OF_HOST_MEMORY: {
        THROW_EXCEPTION(section, "CL_OUT_OF_HOST_MEMORY")
    }
    case CL_PROFILING_INFO_NOT_AVAILABLE: {
        THROW_EXCEPTION(section, "CL_PROFILING_INFO_NOT_AVAILABLE")
    }
    case CL_MEM_COPY_OVERLAP: {
        THROW_EXCEPTION(section, "CL_MEM_COPY_OVERLAP")
    }
    case CL_IMAGE_FORMAT_MISMATCH: {
        THROW_EXCEPTION(section, "CL_IMAGE_FORMAT_MISMATCH")
    }
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: {
        THROW_EXCEPTION(section, "CL_IMAGE_FORMAT_NOT_SUPPORTED")
    }
    case CL_BUILD_PROGRAM_FAILURE: {
        THROW_EXCEPTION(section, "CL_BUILD_PROGRAM_FAILURE")
    }
    case CL_MAP_FAILURE: {
        THROW_EXCEPTION(section, "CL_MAP_FAILURE")
    }
    case CL_MISALIGNED_SUB_BUFFER_OFFSET: {
        THROW_EXCEPTION(section, "CL_MISALIGNED_SUB_BUFFER_OFFSET")
    }
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: {
        THROW_EXCEPTION(section, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST")
    }
    case CL_COMPILE_PROGRAM_FAILURE: {
        THROW_EXCEPTION(section, "CL_COMPILE_PROGRAM_FAILURE")
    }
    case CL_LINKER_NOT_AVAILABLE: {
        THROW_EXCEPTION(section, "CL_LINKER_NOT_AVAILABLE")
    }
    case CL_LINK_PROGRAM_FAILURE: {
        THROW_EXCEPTION(section, "CL_LINK_PROGRAM_FAILURE")
    }
    case CL_DEVICE_PARTITION_FAILED: {
        THROW_EXCEPTION(section, "CL_DEVICE_PARTITION_FAILED")
    }
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: {
        THROW_EXCEPTION(section, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE")
    }
    case CL_INVALID_VALUE: {
        THROW_EXCEPTION(section, "CL_INVALID_VALUE")
    }
    case CL_INVALID_DEVICE_TYPE: {
        THROW_EXCEPTION(section, "CL_INVALID_DEVICE_TYPE")
    }
    case CL_INVALID_PLATFORM: {
        THROW_EXCEPTION(section, "CL_INVALID_PLATFORM")
    }
    case CL_INVALID_DEVICE: {
        THROW_EXCEPTION(section, "CL_INVALID_DEVICE")
            break;
    }
    case CL_INVALID_CONTEXT: {
        THROW_EXCEPTION(section, "CL_INVALID_CONTEXT")
            break;
    }
    case CL_INVALID_QUEUE_PROPERTIES: {
        THROW_EXCEPTION(section, "CL_INVALID_QUEUE_PROPERTIES")
            break;
    }
    case CL_INVALID_COMMAND_QUEUE: {
        THROW_EXCEPTION(section, "CL_INVALID_COMMAND_QUEUE")
            break;
    }
    case CL_INVALID_HOST_PTR: {
        THROW_EXCEPTION(section, "CL_INVALID_HOST_PTR")
            break;
    }
    case CL_INVALID_MEM_OBJECT: {
        THROW_EXCEPTION(section, "CL_INVALID_MEM_OBJECT")
            break;
    }
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: {
        THROW_EXCEPTION(section, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR")
            break;
    }
    case CL_INVALID_IMAGE_SIZE: {
        THROW_EXCEPTION(section, "CL_INVALID_IMAGE_SIZE")
            break;
    }
    case CL_INVALID_SAMPLER: {
        THROW_EXCEPTION(section, "CL_INVALID_SAMPLER")
            break;
    }
    case CL_INVALID_BINARY: {
        THROW_EXCEPTION(section, "CL_INVALID_BINARY")
            break;
    }
    case CL_INVALID_BUILD_OPTIONS: {
        THROW_EXCEPTION(section, "CL_INVALID_BUILD_OPTIONS")
            break;
    }
    case CL_INVALID_PROGRAM: {
        THROW_EXCEPTION(section, "CL_INVALID_PROGRAM")
            break;
    }
    case CL_INVALID_PROGRAM_EXECUTABLE: {
        THROW_EXCEPTION(section, "CL_INVALID_PROGRAM_EXECUTABLE")
            break;
    }
    case CL_INVALID_KERNEL_DEFINITION: {
        THROW_EXCEPTION(section, "CL_INVALID_KERNEL_DEFINITION")
            break;
    }
    case CL_INVALID_KERNEL: {
        THROW_EXCEPTION(section, "CL_INVALID_KERNEL")
            break;
    }
    case CL_INVALID_ARG_INDEX: {
        THROW_EXCEPTION(section, "CL_INVALID_ARG_INDEX")
            break;
    }
    case CL_INVALID_ARG_VALUE: {
        THROW_EXCEPTION(section, "CL_INVALID_ARG_VALUE")
            break;
    }
    case CL_INVALID_ARG_SIZE: {
        THROW_EXCEPTION(section, "CL_INVALID_ARG_SIZE")
            break;
    }
    case CL_INVALID_KERNEL_ARGS: {
        THROW_EXCEPTION(section, "CL_INVALID_KERNEL_ARGS")
            break;
    }
    case CL_INVALID_WORK_DIMENSION: {
        THROW_EXCEPTION(section, "CL_INVALID_WORK_DIMENSION")
            break;
    }
    case CL_INVALID_WORK_GROUP_SIZE: {
        THROW_EXCEPTION(section, "CL_INVALID_WORK_GROUP_SIZE")
            break;
    }
    case CL_INVALID_WORK_ITEM_SIZE: {
        THROW_EXCEPTION(section, "CL_INVALID_WORK_ITEM_SIZE")
            break;
    }
    case CL_INVALID_GLOBAL_OFFSET: {
        THROW_EXCEPTION(section, "CL_INVALID_GLOBAL_OFFSET")
            break;
    }
    case CL_INVALID_EVENT_WAIT_LIST: {
        THROW_EXCEPTION(section, "CL_INVALID_EVENT_WAIT_LIST")
            break;
    }
    case CL_INVALID_EVENT: {
        THROW_EXCEPTION(section, "CL_INVALID_EVENT")
            break;
    }
    case CL_INVALID_OPERATION: {
        THROW_EXCEPTION(section, "CL_INVALID_OPERATION")
            break;
    }
    case CL_INVALID_GL_OBJECT: {
        THROW_EXCEPTION(section, "CL_INVALID_GL_OBJECT")
            break;
    }
    case CL_INVALID_BUFFER_SIZE: {
        THROW_EXCEPTION(section, "CL_INVALID_BUFFER_SIZE")
            break;
    }
    case CL_INVALID_MIP_LEVEL: {
        THROW_EXCEPTION(section, "CL_INVALID_MIP_LEVEL")
            break;
    }
    case CL_INVALID_GLOBAL_WORK_SIZE: {
        THROW_EXCEPTION(section, "CL_INVALID_GLOBAL_WORK_SIZE")
            break;
    }
    case CL_INVALID_PROPERTY: {
        THROW_EXCEPTION(section, "CL_INVALID_PROPERTY")
            break;
    }
    case CL_INVALID_IMAGE_DESCRIPTOR: {
        THROW_EXCEPTION(section, "CL_INVALID_IMAGE_DESCRIPTOR")
            break;
    }
    case CL_INVALID_COMPILER_OPTIONS: {
        THROW_EXCEPTION(section, "CL_INVALID_COMPILER_OPTIONS")
            break;
    }
    case CL_INVALID_LINKER_OPTIONS: {
        THROW_EXCEPTION(section, "CL_INVALID_LINKER_OPTIONS")
            break;
    }
    case CL_INVALID_DEVICE_PARTITION_COUNT: {
        THROW_EXCEPTION(section, "CL_INVALID_DEVICE_PARTITION_COUNT")
            break;
    }
    case CL_INVALID_KERNEL_NAME: {
        THROW_EXCEPTION(section, "CL_INVALID_KERNEL_NAME")
            break;
    }
    default: {
        THROW_EXCEPTION(section, std::to_string(errcode_ret))
    }
    }
}
