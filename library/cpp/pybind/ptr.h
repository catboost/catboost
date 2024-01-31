#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <util/generic/ptr.h>

namespace NPyBind {
    template <class T>
    class TPythonIntrusivePtrOps {
    public:
        static inline void Ref(T* t) noexcept {
            Py_XINCREF(t);
        }

        static inline void UnRef(T* t) noexcept {
#ifdef Py_DEBUG
            if (!Py_IsInitialized()) {
                return;
            }
#endif
            Py_XDECREF(t);
        }

        static inline void DecRef(T* t) noexcept {
            Py_XDECREF(t);
        }
    };

    class TPyObjectPtr: public TIntrusivePtr<PyObject, TPythonIntrusivePtrOps<PyObject>> {
    private:
        typedef TIntrusivePtr<PyObject, TPythonIntrusivePtrOps<PyObject>> TParent;
        typedef TPythonIntrusivePtrOps<PyObject> TOps;

    public:
        inline TPyObjectPtr() noexcept {
        }

        inline explicit TPyObjectPtr(PyObject* obj) noexcept
            : TParent(obj)
        {
        }

        inline TPyObjectPtr(PyObject* obj, bool unref) noexcept
            : TParent(obj)
        {
            if (unref)
                TOps::UnRef(TParent::Get());
        }

        inline PyObject* RefGet() {
            TOps::Ref(TParent::Get());
            return TParent::Get();
        }
    };

}
