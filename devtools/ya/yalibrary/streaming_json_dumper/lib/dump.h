#pragma once

#include <Python.h>

namespace NSJson {
    void DumpToStream(PyObject* obj, PyObject* stream);
}
