#pragma once

#include <Python.h>

PyObject* LoadJsonFromString(const char* data, size_t len, bool internKeys = false, bool internVals = false, bool mayUnicode = false);
