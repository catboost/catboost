#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "ptr.h"
#include "cast.h"
#include "exceptions.h"

#include <util/generic/function.h>

namespace NPyBind {
#if PY_MAJOR_VERSION >= 3
    namespace NPrivate {
        using TFinalizationCallBack = std::function<void()>;
        void AddFinalizationCallBack(TFinalizationCallBack);
        class TAtExitRegistrar: private TNonCopyable {
            TAtExitRegistrar(TPyObjectPtr module);
        public:
            static void Instantiate(TPyObjectPtr module) {
                static TAtExitRegistrar registrar(module);
                Y_UNUSED(registrar);
            }
        };

        class TPyBindModuleRegistrar: private TNonCopyable {
            TPyBindModuleRegistrar();
            TPyObjectPtr Module;
        public:
            static void Instantiate() {
                static TPyBindModuleRegistrar registrar;
                Y_UNUSED(registrar);
            }
        };
    } //NPrivate
#endif

    class TModuleHolder {
    private:
        TModuleHolder(const TModuleHolder&);
        TModuleHolder& operator=(const TModuleHolder&);

        TModuleHolder();
    private:
        typedef PyCFunction TModuleMethod;
        typedef PyCFunctionWithKeywords TModuleMethodWithKeywords;
#if PY_MAJOR_VERSION >= 3
        typedef PyObject* (*TModuleInitFunc)();
#else
        typedef void (*TModuleInitFunc)();
#endif

        struct TMethodDef {
            TString Name;
            TModuleMethod Method;
            TString Description;
            int Flags;

            TMethodDef(const TString& name, TModuleMethod method, const TString& descr, int flags)
                : Name(name)
                , Method(method)
                , Description(descr)
                , Flags(flags)
            {
            }

            operator PyMethodDef() const {
                PyMethodDef cur = {Name.c_str(), Method, Flags, Description.c_str()};
                return cur;
            }
        };

        typedef TSimpleSharedPtr<TVector<TMethodDef>> TMethodDefVecPtr;
        typedef TSimpleSharedPtr<TVector<PyMethodDef>> TPyMethodDefVecPtr;

        TVector<TMethodDefVecPtr> Methods;
        TVector<TPyMethodDefVecPtr> Defs;
#if PY_MAJOR_VERSION >= 3
        //because the md_name will leak otherwise
        class TPyModuleDefWithName {
            PyModuleDef Def;
            TString Name;
        public:
            explicit TPyModuleDefWithName(TString name, TPyMethodDefVecPtr moduleDefs)
                : Name(std::move(name))
            {
                Def = PyModuleDef{
                    PyModuleDef_HEAD_INIT,
                    Name.c_str(),
                    nullptr,
                    -1,
                    moduleDefs->data(),
                    nullptr, nullptr, nullptr, nullptr
                };
            }
            PyModuleDef* GetDefPtr() {
                return &Def;
            }

        };
        TVector<TSimpleSharedPtr<TPyModuleDefWithName>> ModuleDefs;
#endif

        template <typename TFunction>
        static PyObject* FunctionWrapper(TFunction&& func) {
            try {
                PyObject* res = func();
                if (!res && !PyErr_Occurred())
                    ythrow yexception() << "\nModule method exited with NULL, but didn't set Error.\n Options:\n -- Return correct value or None;\n -- Set python exception;\n -- Throw c++ exception.";
                return res;
            } catch (const TPyNativeErrorException&) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_RuntimeError, "Some PY error occurred while trying to call module method (py error was expected to be set, but something went wrong).");
                }
            } catch (const std::exception& ex) {
                PyErr_SetString(TExceptionsHolder::Instance().ToPyException(ex).Get(), ex.what());
            } catch (...) {
                PyErr_SetString(PyExc_RuntimeError, "Unknown error occurred while trying to call module method");
            }
            return nullptr;
        }

        template <TModuleMethod method>
        static PyObject* MethodWrapper(PyObject* obj, PyObject* args) {
            return FunctionWrapper([=] { return method(obj, args); });
        }

        template <TModuleMethodWithKeywords method>
        static PyObject* MethodWithKeywordsWrapper(PyObject* obj, PyObject* args, PyObject* kwargs) {
            return FunctionWrapper([=] { return method(obj, args, kwargs); });
        }

    public:
        static TModuleHolder& Instance() {
            static TModuleHolder Holder;
            return Holder;
        }

        void ImportModule(TPyObjectPtr module, const char* const name, TModuleInitFunc initFunc) {
            PyImport_AppendInittab(const_cast<char*>(name), initFunc);
            TPyObjectPtr importedModule(PyImport_ImportModule(name), true);
            PyModule_AddObject(module.Get(), name, importedModule.Get());
        }

        template <TModuleMethod method>
        void AddModuleMethod(const TString& name, const TString& descr = "") {
            Methods.back()->push_back(TMethodDef(name, MethodWrapper<method>, descr, METH_VARARGS));
        }

        template <TModuleMethodWithKeywords method>
        void AddModuleMethod(const TString& name, const TString& descr = "") {
            Methods.back()->push_back(TMethodDef(name, (TModuleMethod)&MethodWithKeywordsWrapper<method>, descr, METH_VARARGS | METH_KEYWORDS));
        }

        TPyObjectPtr InitModule(const TString& name) {
            Defs.push_back(new TVector<PyMethodDef>(Methods.back()->begin(), Methods.back()->end()));
            PyMethodDef blank = {nullptr, nullptr, 0, nullptr};
            Defs.back()->push_back(blank);
#if PY_MAJOR_VERSION >= 3
            ModuleDefs.push_back(MakeSimpleShared<TPyModuleDefWithName>(name, Defs.back()));
            TPyObjectPtr res(PyModule_Create(ModuleDefs.back()->GetDefPtr()));
            NPrivate::TAtExitRegistrar::Instantiate(res);
            NPrivate::TPyBindModuleRegistrar::Instantiate();
#else
            TPyObjectPtr res(Py_InitModule(name.c_str(), &(Defs.back()->at(0))));
#endif
            Methods.push_back(new TVector<TMethodDef>);
            return res;
        }
    };

    template <typename TMethodSignature, TMethodSignature method>
    class TModuleMethodCaller {
    private:
        template <typename TResult, typename... Args>
        struct TCaller {
            static PyObject* Call(PyObject* args) {
                return BuildPyObject(std::apply(method, GetArguments<Args...>(args)));
            }
        };

        template <typename TResult, typename... Args>
        static PyObject* InternalCall(TResult (*)(Args...), PyObject* args) {
            return BuildPyObject(std::apply(method, GetArguments<Args...>(args)));
        }

    public:
        static PyObject* Call(PyObject*, PyObject* args) {
            if (args && (!PyTuple_Check(args) || PyTuple_Size(args) != TFunctionArgs<TMethodSignature>::Length)) {
                ythrow yexception() << "Method takes " << (size_t)(TFunctionArgs<TMethodSignature>::Length) << " arguments, " << PyTuple_Size(args) << " provided";
            }

            return InternalCall(method, args);
        }
    };

}
