#ifndef _GPU_EXCEPTIONS_H
#define _GPU_EXCEPTIONS_H

#include <iostream>
#include <string>
#include <exception>
#include <CL/cl.h>

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

void errorProcessing(const std::string& section, const cl_int errcode_ret);

#endif //_GPU_EXCEPTIONS_H
