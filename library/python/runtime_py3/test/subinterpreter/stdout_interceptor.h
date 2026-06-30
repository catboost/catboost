#pragma once

#include <Python.h>

class IOutputStream;

class TPyStdoutInterceptor {
public:
    TPyStdoutInterceptor(IOutputStream& redirectionStream) noexcept;
    ~TPyStdoutInterceptor() noexcept;

    static bool SetupInterceptionSupport() noexcept;

private:
    PyObject* RealStdout_;
};
