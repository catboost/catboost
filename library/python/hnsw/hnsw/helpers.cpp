#include <Python.h>

#include "helpers.h"

#include <library/cpp/hnsw/helpers/interrupt.h>
#include <util/memory/blob.h>

namespace NHnsw::PythonHelpers {
    void PyCheckInterrupted() {
        TGilGuard guard;
        if (PyErr_CheckSignals() == -1) {
            throw TInterruptException();
        }
    }

    void SetPythonInterruptHandler() {
        SetInterruptHandler(PyCheckInterrupted);
    }

    void ResetPythonInterruptHandler() {
        ResetInterruptHandler();
    }

    template <>
    const char* NumpyTypeDescription<i32>() {
        return "int32";
    }

    template <>
    const char* NumpyTypeDescription<ui32>() {
        return "uint32";
    }

    template <>
    const char* NumpyTypeDescription<i64>() {
        return "int64";
    }

    template <>
    const char* NumpyTypeDescription<ui64>() {
        return "uint64";
    }

    template <>
    const char* NumpyTypeDescription<float>() {
        return "float32";
    }

    template <>
    const char* NumpyTypeDescription<double>() {
        return "float64";
    }

    template <>
    PyObject* ToPyObject<i32>(i32 value) {
        return PyLong_FromLong(value);
    }

    template <>
    PyObject* ToPyObject<ui32>(ui32 value) {
        return PyLong_FromUnsignedLong(value);
    }

    template <>
    PyObject* ToPyObject<i64>(i64 value) {
        return PyLong_FromLongLong(value);
    }

    template <>
    PyObject* ToPyObject<ui64>(ui64 value) {
        return PyLong_FromUnsignedLongLong(value);
    }

    void SaveIndex(const TBlob& indexBlob, const TString& indexPath) {
        TOFStream out(indexPath);
        out.Write(indexBlob.Begin(), indexBlob.Size());
        out.Finish();
    }

    TBlob LoadIndex(const TString& indexPath) {
        return TBlob::PrechargedFromFile(indexPath);
    }

}
