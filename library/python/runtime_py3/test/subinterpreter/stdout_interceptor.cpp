#include "stdout_interceptor.h"

#include <util/stream/output.h>

namespace {

struct TOStreamWrapper {
    PyObject_HEAD
    IOutputStream* Stm = nullptr;
};

PyObject* Write(TOStreamWrapper *self, PyObject *const *args, Py_ssize_t nargs) noexcept {
    try {
        Py_buffer view;
        for (Py_ssize_t i = 0; i < nargs; ++i) {
            PyObject* buf = args[i];
            if (PyUnicode_Check(args[i])) {
                buf = PyUnicode_AsUTF8String(buf);
                if (!buf) {
                    return nullptr;
                }
            }

            if (PyObject_GetBuffer(buf, &view, PyBUF_SIMPLE | PyBUF_C_CONTIGUOUS) == -1) {
                return nullptr;
            }
            self->Stm->Write(reinterpret_cast<const char*>(view.buf), view.len);
            PyBuffer_Release(&view);
        }

        return Py_None;
    } catch(const std::exception& err) {
        PyErr_SetString(PyExc_IOError, err.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unhandled C++ exception of unknown type");
    }
    return nullptr;
}

PyMethodDef TOStreamWrapperMethods[] = {
    {"write", reinterpret_cast<PyCFunction>(Write), METH_FASTCALL, PyDoc_STR("write buffer to wrapped C++ stream")},
    {}
};

PyTypeObject TOStreamWrapperType {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "testwrap.OStream",
    .tp_basicsize = sizeof(TOStreamWrapper),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("C++ IOStream wrapper"),
    .tp_methods = TOStreamWrapperMethods,
    .tp_new = PyType_GenericNew,
};

}

TPyStdoutInterceptor::TPyStdoutInterceptor(IOutputStream& redirectionStream) noexcept
    : RealStdout_{PySys_GetObject("stdout")}
{
    Py_INCREF(RealStdout_);

    PyObject* redirect = TOStreamWrapperType.tp_alloc(&TOStreamWrapperType, 0);
    reinterpret_cast<TOStreamWrapper*>(redirect)->Stm = &redirectionStream;

    PySys_SetObject("stdout", redirect);
    Py_DECREF(redirect);
}

TPyStdoutInterceptor::~TPyStdoutInterceptor() noexcept {
    PySys_SetObject("stdout", RealStdout_);
    Py_DECREF(RealStdout_);
}

bool TPyStdoutInterceptor::SetupInterceptionSupport() noexcept {
    return PyType_Ready(&TOStreamWrapperType) == 0;
}
