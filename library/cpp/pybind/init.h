#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "ptr.h"

namespace NPyBind {
#if PY_MAJOR_VERSION >= 3

#define PYBIND_MODINIT(name) PyMODINIT_FUNC PyInit_##name()

    inline PyObject* ModInitReturn(TPyObjectPtr&& modptr) {
        return modptr.Release();
    }

#else

#define PYBIND_MODINIT(name) PyMODINIT_FUNC init##name()

    inline void ModInitReturn(TPyObjectPtr&&) {
    }

#endif
}
