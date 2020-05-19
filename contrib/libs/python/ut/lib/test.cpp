#include "test.h"

#include <Python.h>
#include <library/cpp/unittest/registar.h>

TTestPyInvoker::TTestPyInvoker() {}

const char* TTestPyInvoker::GetVersion() {
    Py_Initialize();

    auto* module = PyImport_ImportModule("sys");
    UNIT_ASSERT(module != nullptr);

    auto* versionObj = PyObject_GetAttrString(module, "version");
    if (versionObj == nullptr) {
        Py_DECREF(module);
        UNIT_ASSERT(versionObj != nullptr);
    }

    return Py_GetVersion();
}
