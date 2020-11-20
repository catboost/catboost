#pragma once

#include <Python.h>

#include <library/cpp/yson/node/node.h>

namespace NYT {
    PyObject* BuildPyObject(const TNode& val);
}
