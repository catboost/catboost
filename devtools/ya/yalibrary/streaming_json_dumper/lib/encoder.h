#pragma once

#include <Python.h>

#include <util/generic/yexception.h>

namespace NSJson {

    struct TValueError : public yexception {
    };

    void Encode(PyObject* obj, PyObject* stream);
}
