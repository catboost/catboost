#include "module.h"
#include "ptr.h"

#include <util/generic/adaptor.h>

namespace NPyBind {

#if PY_MAJOR_VERSION >= 3
    namespace NPrivate {
        struct TFinCallBacksHolder {
            static TVector<TFinalizationCallBack>& GetCallBacks() {
                static TVector<TFinalizationCallBack> res;
                return res;
            }
        };

        TAtExitRegistrar::TAtExitRegistrar(TPyObjectPtr module) {
            TPyObjectPtr atExitModuleName(Py_BuildValue("s", "atexit"), true);
            TPyObjectPtr atExitModule(PyImport_Import(atExitModuleName.Get()));
            Y_ABORT_UNLESS(atExitModule);
            TPyObjectPtr finalizerFunc(PyObject_GetAttrString(module.Get(), "finalizer"), true);
            Y_ABORT_UNLESS(finalizerFunc);
            TPyObjectPtr registerName(Py_BuildValue("s", "register"), true);
            PyObject_CallMethodObjArgs(atExitModule.Get(), registerName.Get(), finalizerFunc.Get(), nullptr);
        }

        TPyBindModuleRegistrar::TPyBindModuleRegistrar() {
            TPyObjectPtr modules(PySys_GetObject("modules"));
            Y_ENSURE(modules.Get());
            if (Module = NPrivate::CreatePyBindModule()) {
                Y_ABORT_UNLESS(0 == PyDict_SetItemString(modules.Get(), "pybind", Module.RefGet()));
            }
            AddFinalizationCallBack([this]() {
                auto ptr = Module;
                Y_UNUSED(ptr);
                TPyObjectPtr modules(PySys_GetObject("modules"));
                Y_ENSURE(modules.Get());
                TPyObjectPtr pyBindName(Py_BuildValue("s", "pybind"));
                if (PyDict_Contains(modules.Get(), pyBindName.Get()) == 1) {
                    Y_ABORT_UNLESS(0==PyDict_DelItemString(modules.Get(), "pybind"));
                }
                if (Module) {
                    //We have to untrack the module because some refs from him refers to gc-leaked errors
                    //see exceptions.cpp fore more info
                    PyObject_GC_UnTrack(Module.Get());
                    Module.Drop();
                }
            });
        }

        void AddFinalizationCallBack(TFinalizationCallBack callback) {
            TFinCallBacksHolder::GetCallBacks().push_back(callback);
        }

        int FinalizeAll() {
            for (auto callback: Reversed(NPrivate::TFinCallBacksHolder::GetCallBacks())) {
                callback();
            }
            return 0;
        }
    }
#endif


    TModuleHolder::TModuleHolder()
        : Methods(1, new TVector<TMethodDef>)
    {
#if PY_MAJOR_VERSION >= 3
        AddModuleMethod<TModuleMethodCaller<decltype(&NPrivate::FinalizeAll), &NPrivate::FinalizeAll>::Call>("finalizer");
#endif
    }
}//NPyBind
