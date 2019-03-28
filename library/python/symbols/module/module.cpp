#include <Python.h>

#include <library/python/symbols/registry/syms.h>

#include <util/generic/string.h>

#define CAP(x) SYM_2(x, x)

BEGIN_SYMS("_capability")
#if defined(_musl_)
CAP("musl")
#endif
#if defined(_linux_)
CAP("linux")
#endif
#if defined(_darwin_)
CAP("darwin")
#endif
CAP("_sentinel")
END_SYMS()

#undef CAP

using namespace NPrivate;

namespace {
    template <class T>
    struct TCB: public ICB {
        inline TCB(T& t)
            : CB(&t)
        {
        }

        void Apply(const char* mod, const char* name, void* sym) override {
            (*CB)(mod, name, sym);
        }

        T* CB;
    };

    template <class T>
    static inline TCB<T> MakeTCB(T& t) {
        return t;
    }
}

static void DictSetStringPtr(PyObject* dict, const char* name, void* value) {
    PyObject* p = PyLong_FromVoidPtr(value);
    PyDict_SetItemString(dict, name, p);
    Py_DECREF(p);
}

static PyObject* InitSyms(PyObject* m) {
    if (!m)
        return NULL;
    PyObject* d = PyDict_New();
    if (!d)
        return NULL;

    auto f = [&](const char* mod, const char* name, void* sym) {
        DictSetStringPtr(d, (TString(mod) + "|" + TString(name)).c_str(), sym);
    };

    auto cb = MakeTCB(f);

    ForEachSymbol(cb);

    if (PyObject_SetAttrString(m, "syms", d))
        m = NULL;
    Py_DECREF(d);
    return m;
}

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "syms", NULL, -1, NULL, NULL, NULL, NULL, NULL};

extern "C" PyObject* PyInit_syms() {
    return InitSyms(PyModule_Create(&module));
}
#else
extern "C" void initsyms() {
    InitSyms(Py_InitModule("syms", NULL));
}
#endif
