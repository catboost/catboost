#pragma once

#include <library/cpp/yson/node/node.h>

#include <Python.h>

namespace NYT {
    PyObject* BuildPyObject(const TNode& val);
}
