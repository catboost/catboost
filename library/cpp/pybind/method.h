#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <util/generic/string.h>
#include <util/generic/map.h>
#include <util/generic/set.h>
#include <util/generic/vector.h>
#include <util/generic/ptr.h>
#include <util/generic/typetraits.h>

#include <util/generic/function.h>

#include "cast.h"
#include "exceptions.h"

namespace NPyBind {
    template <typename TObjType>
    class TBaseMethodCaller {
    public:
        virtual ~TBaseMethodCaller() {
        }
        virtual bool CallMethod(PyObject* owner, TObjType* self, PyObject* args, PyObject* kwargs, PyObject*& res) const = 0;
        virtual bool HasMethod(PyObject*, TObjType*, const TString&, const TSet<TString>&) {
            return true;
        }
    };

    template <typename TObjType>
    class TIsACaller;

    template <typename TObjType>
    class TMethodCallers {
    private:
        typedef TSimpleSharedPtr<TBaseMethodCaller<TObjType>> TCallerPtr;
        typedef TVector<TCallerPtr> TCallerList;
        typedef TMap<TString, TCallerList> TCallerMap;

        const TSet<TString>& HiddenAttrNames;
        TCallerMap Callers;

    public:
        TMethodCallers(const TSet<TString>& hiddenNames)
            : HiddenAttrNames(hiddenNames)
        {
        }

        void AddCaller(const TString& name, TCallerPtr caller) {
            Callers[name].push_back(caller);
        }

        bool HasCaller(const TString& name) const {
            return Callers.has(name);
        }

        PyObject* CallMethod(PyObject* owner, TObjType* self, PyObject* args, PyObject* kwargs, const TString& name) const {
            const TCallerList* lst = Callers.FindPtr(name);
            if (!lst)
                return nullptr;
            for (const auto& caller : *lst) {
                PyObject* res = nullptr;
                PyErr_Clear();
                if (caller->CallMethod(owner, self, args, kwargs, res))
                    return res;
            }
            return nullptr;
        }

        bool HasMethod(PyObject* owner, TObjType* self, const TString& name) const {
            const TCallerList* lst = Callers.FindPtr(name);
            if (!lst)
                return false;
            for (const auto& caller : *lst) {
                if (caller->HasMethod(owner, self, name, HiddenAttrNames))
                    return true;
            }
            return false;
        }

        void GetMethodsNames(PyObject* owner, TObjType* self, TVector<TString>& resultNames) const {
            for (const auto& it : Callers) {
                if (HasMethod(owner, self, it.first) && !HiddenAttrNames.contains(it.first))
                    resultNames.push_back(it.first);
            }
        }

        void GetAllMethodsNames(TVector<TString>& resultNames) const {
            for (const auto& it : Callers) {
                resultNames.push_back(it.first);
            }
        }

        void GetPropertiesNames(PyObject*, TObjType* self, TVector<TString>& resultNames) const {
            const TCallerList* lst = Callers.FindPtr("IsA");
            if (!lst)
                return;
            for (const auto& caller : *lst) {
                TIsACaller<TObjType>* isACaller = dynamic_cast<TIsACaller<TObjType>*>(caller.Get());
                if (isACaller) {
                    resultNames = isACaller->GetPropertiesNames(self);
                    return;
                }
            }
        }
    };

    template <typename TObjType>
    class TIsACaller: public TBaseMethodCaller<TObjType> {
    private:
        class TIsAChecker {
        public:
            virtual ~TIsAChecker() {
            }
            virtual bool Check(const TObjType* obj) const = 0;
        };

        template <typename TConcrete>
        class TIsAConcreteChecker: public TIsAChecker {
        public:
            bool Check(const TObjType* obj) const override {
                return dynamic_cast<const TConcrete*>(obj) != nullptr;
            }
        };

        typedef TSimpleSharedPtr<TIsAChecker> TCheckerPtr;
        typedef TMap<TString, TCheckerPtr> TCheckersMap;

        TCheckersMap Checkers;

        bool Check(const TString& name, const TObjType* obj) const {
            const TCheckerPtr* checker = Checkers.FindPtr(name);
            if (!checker) {
                PyErr_Format(PyExc_KeyError, "unknown class name: %s", name.data());
                return false;
            }
            return (*checker)->Check(obj);
        }

    protected:
        TIsACaller() {
        }

        template <typename TConcrete>
        void AddChecker(const TString& name) {
            Checkers[name] = new TIsAConcreteChecker<TConcrete>;
        }

    public:
        bool CallMethod(PyObject*, TObjType* self, PyObject* args, PyObject*, PyObject*& res) const override {
            if (args == nullptr || !PyTuple_Check(args))
                return false;
            size_t cnt = PyTuple_Size(args);
            bool result = true;
            for (size_t i = 0; i < cnt; ++i) {
                result = result && Check(
#if PY_MAJOR_VERSION >= 3
                        PyUnicode_AsUTF8(
#else
                        PyString_AsString(
#endif
                            PyTuple_GetItem(args, i)), self);
            }
            if (PyErr_Occurred()) {
                return false;
            }
            res = BuildPyObject(result);
            return true;
        }

        TVector<TString> GetPropertiesNames(const TObjType* obj) const {
            TVector<TString> names;

            for (const auto& it : Checkers) {
                if (it.second->Check(obj)) {
                    names.push_back(it.first);
                }
            }

            return names;
        }
    };

    template <typename TObjType>
    class TGenericMethodCaller: public TBaseMethodCaller<TObjType> {
    private:
        TString AttrName;

    public:
        TGenericMethodCaller(const TString& attrName)
            : AttrName(attrName)
        {
        }

        bool CallMethod(PyObject* obj, TObjType*, PyObject* args, PyObject*, PyObject*& res) const override {
            auto str = NameFromString(AttrName);
            PyObject* attr = PyObject_GenericGetAttr(obj, str.Get());
            if (!attr)
                ythrow yexception() << "Can't get generic attribute '" << AttrName << "'";
            res = PyObject_CallObject(attr, args);
            return res != nullptr;
        }
    };


    template <typename TObjType, typename TSubObject>
    class TSubObjectChecker: public TBaseMethodCaller<TObjType> {
    public:
        ~TSubObjectChecker() override {
        }

        bool HasMethod(PyObject*, TObjType* self, const TString&, const TSet<TString>&) override {
            return dynamic_cast<const TSubObject*>(self) != nullptr;
        }
    };

    template <typename TFunctor, typename Tuple, typename ResType, typename=std::enable_if_t<!std::is_same_v<ResType, void>>>
    void ApplyFunctor(TFunctor functor, Tuple resultArgs, PyObject*& res) {
        res = BuildPyObject(std::move(Apply(functor, resultArgs)));
    }

    template <typename TFunctor, typename Tuple, typename ResType, typename=std::enable_if_t<std::is_same_v<ResType, void>>, typename=void>
    void ApplyFunctor(TFunctor functor, Tuple resultArgs, PyObject*& res) {
        Py_INCREF(Py_None);
        res = Py_None;
        Apply(functor, resultArgs);
    }

    template <typename TObjType, typename TResType, typename... Args>
    class TFunctorCaller: public TBaseMethodCaller<TObjType> {
        using TFunctor = std::function<TResType(TObjType&,Args...)>;
        TFunctor Functor;
    public:
        explicit TFunctorCaller(TFunctor functor):
            Functor(functor){}

        bool CallMethod(PyObject*, TObjType* self, PyObject* args, PyObject*, PyObject*& res) const {
            auto methodArgsTuple = GetArguments<Args...>(args);
            auto resultArgs = std::tuple_cat(std::tie(*self), methodArgsTuple);
            ApplyFunctor<TFunctor, decltype(resultArgs), TResType>(Functor, resultArgs, res);
            return true;
        }
    };

    template <typename TObjType, typename TRealType>
    class TGetStateCaller: public TSubObjectChecker<TObjType, TRealType> {
    protected:
        TPyObjectPtr AddFromCaller(PyObject* obj, const TString& methodName) const {
            PyObject* res = PyObject_CallMethod(obj, const_cast<char*>(methodName.c_str()), const_cast<char*>(""));
            if (!res) {
                PyErr_Clear();
                return TPyObjectPtr(Py_None);
            }
            return TPyObjectPtr(res, true);
        }

        void GetStandartAttrsDictionary(PyObject* obj, TRealType*, TMap<TString, TPyObjectPtr>& dict) const {
            TPyObjectPtr attrsDict(PyObject_GetAttrString(obj, "__dict__"), true);
            TMap<TString, TPyObjectPtr> attrs;
            if (!FromPyObject(attrsDict.Get(), attrs))
                ythrow yexception() << "Can't get '__dict__' attribute";
            dict.insert(attrs.begin(), attrs.end());
        }

        virtual void GetAttrsDictionary(PyObject* obj, TRealType* self, TMap<TString, TPyObjectPtr>& dict) const = 0;

    public:
        bool CallMethod(PyObject* obj, TObjType* self, PyObject* args, PyObject*, PyObject*& res) const override {
            if (!ExtractArgs(args))
                ythrow yexception() << "Can't parse arguments: it should be none";
            TRealType* rself = dynamic_cast<TRealType*>(self);
            if (!rself)
                return false;
            TMap<TString, TPyObjectPtr> dict;
            GetAttrsDictionary(obj, rself, dict);
            res = BuildPyObject(dict);
            return true;
        }
    };

    template <typename TObjType, typename TRealType>
    class TSetStateCaller: public TSubObjectChecker<TObjType, TRealType> {
    protected:
        void SetStandartAttrsDictionary(PyObject* obj, TRealType*, TMap<TString, TPyObjectPtr>& dict) const {
            TPyObjectPtr value(BuildPyObject(dict), true);
            PyObject_SetAttrString(obj, "__dict__", value.Get());
        }

        virtual void SetAttrsDictionary(PyObject* obj, TRealType* self, TMap<TString, TPyObjectPtr>& dict) const = 0;

    public:
        bool CallMethod(PyObject* obj, TObjType* self, PyObject* args, PyObject*, PyObject*& res) const override {
            TMap<TString, TPyObjectPtr> dict;
            if (!ExtractArgs(args, dict))
                ythrow yexception() << "Can't parse arguments: it should be one dictionary";
            TRealType* rself = dynamic_cast<TRealType*>(self);
            if (!rself)
                return false;
            SetAttrsDictionary(obj, rself, dict);
            Py_INCREF(Py_None);
            res = Py_None;
            return true;
        }
    };

    template <typename TObjType, typename TResult, typename TSubObject, typename TMethod, typename... Args>
    class TAnyParameterMethodCaller: public TSubObjectChecker<TObjType, TSubObject> {
    private:
        TMethod Method;

    public:
        TAnyParameterMethodCaller(TMethod method)
            : Method(method)
        {
        }

    public:
        bool CallMethod(PyObject*, TObjType* self, PyObject* args, PyObject*, PyObject*& res) const override {
            TSubObject* sub = dynamic_cast<TSubObject*>(self);
            if (sub == nullptr)
                return false;
            if (args && (!PyTuple_Check(args) || PyTuple_Size(args) != TFunctionArgs<TMethod>::Length)) {
                //ythrow yexception() << "Method takes " << (size_t)(TFunctionArgs<TMethod>::Length) << " arguments, " << PyTuple_Size(args) << " provided";
                return false;
            }

            try {
                class Applicant {
                public:
                    TResult operator()(Args... theArgs) {
                        return (Sub->*Method)(theArgs...);
                    }
                    TSubObject* Sub;
                    TMethod Method;
                };
                res = BuildPyObject(std::move(Apply(Applicant{sub, Method}, GetArguments<Args...>(args))));
            } catch (cast_exception) {
                return false;
            } catch (const TPyNativeErrorException&) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_RuntimeError, "Some PY error occurred, but it is not set.");
                }
                return true;
            } catch (...) {
                if (PyExc_StopIteration == PyErr_Occurred()) {
                    // NB: it's replacement for geo_boost::python::throw_error_already_set();
                    return true;
                }
                PyErr_SetString(PyExc_RuntimeError, CurrentExceptionMessage().data());
                return true;
            }

            return true;
        }
    };

    template <typename TObjType, typename TSubObject, typename TMethod, typename... Args>
    class TAnyParameterMethodCaller<TObjType, void, TSubObject, TMethod, Args...>: public TSubObjectChecker<TObjType, TSubObject> {
    private:
        TMethod Method;

    public:
        TAnyParameterMethodCaller(TMethod method)
            : Method(method)
        {
        }

    public:
        bool CallMethod(PyObject*, TObjType* self, PyObject* args, PyObject*, PyObject*& res) const override {
            TSubObject* sub = dynamic_cast<TSubObject*>(self);
            if (sub == nullptr) {
                return false;
            }
            if (args && (!PyTuple_Check(args) || PyTuple_Size(args) != TFunctionArgs<TMethod>::Length)) {
                return false;
            }

            try {
                class Applicant {
                public:
                    void operator()(Args... theArgs) {
                        (Sub->*Method)(theArgs...);
                    }
                    TSubObject* Sub;
                    TMethod Method;
                };

                Apply(Applicant{sub, Method}, GetArguments<Args...>(args));

                Py_INCREF(Py_None);
                res = Py_None;
            } catch (cast_exception) {
                return false;
            } catch (const TPyNativeErrorException&) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_RuntimeError, "Some PY error occurred, but it is not set.");
                }
                return true;
            } catch (...) {
                PyErr_SetString(PyExc_RuntimeError, CurrentExceptionMessage().data());
                return true;
            }

            return true;
        }
    };

    template <typename TResult, typename TSubObject, typename... Args>
    struct TConstTraits {
        typedef TResult (TSubObject::*TMethod)(Args... args) const;
    };

    template <typename TResult, typename TSubObject, typename... Args>
    struct TNonConstTraits {
        typedef TResult (TSubObject::*TMethod)(Args... args);
    };

    template <typename TObjType, typename TResult, typename TSubObject, typename TMethod, typename... Args>
    class TConstMethodCaller: public TAnyParameterMethodCaller<TObjType, TResult, const TSubObject, typename TConstTraits<TResult, TSubObject, Args...>::TMethod, Args...> {
    public:
        TConstMethodCaller(typename TConstTraits<TResult, TSubObject, Args...>::TMethod method)
            : TAnyParameterMethodCaller<TObjType, TResult, const TSubObject, typename TConstTraits<TResult, TSubObject, Args...>::TMethod, Args...>(method)
        {
        }
    };

    template <typename TObjType, typename TResult, typename TSubObject, typename... Args>
    TSimpleSharedPtr<TBaseMethodCaller<TObjType>> CreateConstMethodCaller(TResult (TSubObject::*method)(Args...) const) {
        return new TConstMethodCaller<TObjType, TResult, TSubObject, TResult (TSubObject::*)(Args...) const, Args...>(method);
    }

    template <typename TObjType, typename TResType, typename... Args>
    TSimpleSharedPtr<TBaseMethodCaller<TObjType>> CreateFunctorCaller(std::function<TResType(TObjType&, Args...)> functor) {
        return new TFunctorCaller<TObjType, TResType, Args...>(functor);
    }

    template <typename TObjType, typename TResult, typename TSubObject, typename TMethod, typename... Args>
    class TMethodCaller: public TAnyParameterMethodCaller<TObjType, TResult, TSubObject, typename TNonConstTraits<TResult, TSubObject, Args...>::TMethod, Args...> {
    public:
        TMethodCaller(typename TNonConstTraits<TResult, TSubObject, Args...>::TMethod method)
            : TAnyParameterMethodCaller<TObjType, TResult, TSubObject, typename TNonConstTraits<TResult, TSubObject, Args...>::TMethod, Args...>(method)
        {
        }
    };

    template <typename TObjType, typename TResult, typename TSubObject, typename... Args>
    TSimpleSharedPtr<TBaseMethodCaller<TObjType>> CreateMethodCaller(TResult (TSubObject::*method)(Args...)) {
        return new TMethodCaller<TObjType, TResult, TSubObject, TResult (TSubObject::*)(Args...), Args...>(method);
    }

}
