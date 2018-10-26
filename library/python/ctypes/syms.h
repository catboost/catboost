#pragma once

#include "Python.h"

static PyObject* InitSyms(PyObject*);

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "syms", NULL, -1, NULL, NULL, NULL, NULL, NULL};

PyObject* PyInit_syms() {
    return InitSyms(PyModule_Create(&module));
}
#else
void initsyms() {
    InitSyms(Py_InitModule("syms", NULL));
}
#endif

static void DictSetStringPtr(PyObject* dict, const char* name, void* value) {
    PyObject* p = PyLong_FromVoidPtr(value);
    PyDict_SetItemString(dict, name, p);
    Py_DECREF(p);
}

#define BEGIN_SYMS()                         \
    static PyObject* InitSyms(PyObject* m) { \
        if (!m)                              \
            return NULL;                     \
        PyObject* d = PyDict_New();          \
        if (!d)                              \
            return NULL;

#define SYM(SYM_NAME) \
    DictSetStringPtr(d, #SYM_NAME, &SYM_NAME);

#define ESYM(SYM_NAME)      \
    extern void SYM_NAME(); \
    SYM(SYM_NAME)

#define END_SYMS()                            \
    if (PyObject_SetAttrString(m, "syms", d)) \
        m = NULL;                             \
    Py_DECREF(d);                             \
    return m;                                 \
    }
