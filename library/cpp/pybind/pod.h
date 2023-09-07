#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "attr.h"
#include "typedesc.h"

namespace NPyBind {
    struct TPOD {
        TPyObjectPtr Dict;

        TPOD()
            : Dict(PyDict_New(), true)
        {
        }
        bool SetAttr(const char* name, PyObject* value) {
            return PyDict_SetItemString(Dict.Get(), name, value) == 0;
        }
        PyObject* GetAttr(const char* name) const {
            PyObject* res = PyDict_GetItemString(Dict.Get(), name);
            Py_XINCREF(res);
            return res;
        }
    };

    class TPODTraits: public NPyBind::TPythonType<TPOD, TPOD, TPODTraits> {
    private:
        typedef TPythonType<TPOD, TPOD, TPODTraits> MyParent;
        friend class TPythonType<TPOD, TPOD, TPODTraits>;
        TPODTraits();

    public:
        static TPOD* GetObject(TPOD& obj) {
            return &obj;
        }
    };

    template <>
    inline bool FromPyObject<TPOD*>(PyObject* obj, TPOD*& res) {
        res = TPODTraits::CastToObject(obj);
        if (res == nullptr)
            return false;
        return true;
    }
    template <>
    inline bool FromPyObject<const TPOD*>(PyObject* obj, const TPOD*& res) {
        res = TPODTraits::CastToObject(obj);
        if (res == nullptr)
            return false;
        return true;
    }

}
