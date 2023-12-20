#include "exceptions.h"
#include "cast.h"
#include "module.h"
#include <util/generic/algorithm.h>

namespace NPyBind {

    namespace NPrivate {
        TPyObjectPtr CreatePyBindModule() {
            return TPyObjectPtr(TExceptionsHolder::DoInitPyBindModule(), true);
        }
    }//NPrivate

    TPyObjectPtr TExceptionsHolder::GetException(const TString& name) {
        if (name == "")
            return TPyObjectPtr(nullptr);
        if (!Exceptions[name].Get())
            ythrow yexception() << "Wrong base class '" << name << "'";
        return Exceptions[name];
    }

    TPyObjectPtr TExceptionsHolder::GetExceptions(const TVector<TString>& names) {
        TVector<TString> tmp(names.begin(), names.end());
        TVector<TString>::iterator end = std::unique(tmp.begin(), tmp.end());
        TPyObjectPtr tuple(PyTuple_New(std::distance(tmp.begin(), end)), true);
        for (size_t i = 0; i < (size_t)std::distance(tmp.begin(), end); ++i) {
            if (!Exceptions[tmp[i]].Get())
                ythrow yexception() << "Wrong base class '" << tmp[i] << "'";
            PyTuple_SetItem(tuple.Get(), i, Exceptions[tmp[i]].Get());
        }
        return tuple;
    }

    // def PyBindObjectReconstructor(cl, props):
    //    return cl(__properties__=props)
    static PyObject* PyBindObjectReconstructor(PyObject*, PyObject* args) {
        TPyObjectPtr callable, props;
        if (!ExtractArgs(args, callable, props))
            ythrow yexception() << "Wrong method arguments";
#if PY_MAJOR_VERSION >= 3
        TPyObjectPtr noArgs(PyTuple_New(0), true);
#else
        TPyObjectPtr noArgs(PyList_New(0), true);
#endif
        TPyObjectPtr kw(PyDict_New(), true);
        PyDict_SetItemString(kw.Get(), "__properties__", props.Get());
        TPyObjectPtr res(PyObject_Call(callable.Get(), noArgs.Get(), kw.Get()), true);
        return res.RefGet();
    }

    static PyMethodDef PyBindMethods[] = {
        {"PyBindObjectReconstructor", PyBindObjectReconstructor, METH_VARARGS, "Tech method. It's required for unpickling."},
        {nullptr, nullptr, 0, nullptr}};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
                PyModuleDef_HEAD_INIT,
                "pybind",
                NULL,
                -1,
                PyBindMethods,
                NULL, NULL, NULL, NULL
    };

    static PyObject* InitPyBind() {
        return PyModule_Create(&moduledef);
    }
#else
    static PyObject* InitPyBind() {
        return Py_InitModule("pybind", PyBindMethods);
    }
#endif

    void TExceptionsHolder::DoInitPyBindModule2() {
        DoInitPyBindModule();
    }

    PyObject* TExceptionsHolder::DoInitPyBindModule() {
        Instance().Module = NPyBind::TPyObjectPtr(InitPyBind(), true);
        if (!Instance().Module.Get())
            return nullptr;

        for (TCheckersVector::const_iterator it = Instance().Checkers.begin(), end = Instance().Checkers.end(); it != end; ++it) {
            TString name = (*it)->GetName();
            if (!!name) {
                //Ref to the object should be incremented before passing to AddObject
                auto res = PyModule_AddObject(Instance().Module.Get(), name.data(), (*it)->GetException().RefGet());
                if (res < 0) {
                    ythrow yexception() << "Failed to add object " << name << " to internal module pybind";
                }
            }
        }
        return Instance().Module.RefGet();
    }

    void TExceptionsHolder::Clear() {
        //Unfortunately in Python3 we can't retrack this object because of PyError_NewException
        //it's only the safe way to preserve GC gens in valid state during the finalization
        for (auto& ptr: Checkers) {
            if (!dynamic_cast<const TPyErrExceptionsChecker*>(ptr.Get())) {  // no need to untrack standard PyExc_* exceptions from TPyErrExceptionsChecker
                if (auto exceptionPtr = ptr->GetException()) {
                    PyObject_GC_UnTrack(exceptionPtr.Get());
                }
            }
        }
        Checkers.clear();
        Exceptions.clear();
        Module.Drop();
    }

    TExceptionsHolder::TExceptionsHolder() {
        AddException<std::exception>("yexception");
        AddException<TSystemError>("TSystemError", "yexception");
        AddException<TIoException>("TIoException", "yexception");

        TVector<TString> names(2);
        names[0] = "TSystemError";
        names[1] = "TIoException";

        AddException<TIoSystemError>("TIoSystemError", names);
        AddException<TFileError>("TFileError", "TIoSystemError");
        AddException<TBadCastException>("TBadCastException", "yexception");

        Checkers.push_back(new TPyErrExceptionsChecker);

#if PY_MAJOR_VERSION >= 3
        NPrivate::AddFinalizationCallBack([this]() {
            Clear();
        });
#else
        PyImport_AppendInittab("pybind", DoInitPyBindModule2);
#endif
    }

    NPyBind::TPyObjectPtr TExceptionsHolder::ToPyException(const std::exception& ex) {
        for (TCheckersVector::const_reverse_iterator it = Checkers.rbegin(), end = Checkers.rend(); it != end; ++it) {
            if ((*it)->Check(ex))
                return (*it)->GetException();
        }
        return TPyObjectPtr(nullptr);
    }
}
