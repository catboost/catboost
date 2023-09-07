#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <util/generic/string.h>
#include <util/generic/map.h>
#include <util/generic/set.h>
#include <util/generic/vector.h>
#include <util/generic/ptr.h>

#include "cast.h"
#include "exceptions.h"

namespace NPyBind {
    // TBaseAttrGetter
    template <typename TObjType>
    class TBaseAttrGetter {
    public:
        virtual ~TBaseAttrGetter() {
        }
        virtual bool GetAttr(PyObject* owner, const TObjType& self, const TString& attr, PyObject*& res) const = 0;

        virtual bool HasAttr(PyObject* owner, const TObjType& self, const TString& attr, const TSet<TString>& hiddenNames) const {
            if (hiddenNames.find(attr) != hiddenNames.end())
                return false;
            PyObject* res = nullptr;
            if (!GetAttr(owner, self, attr, res))
                return false;
            Py_XDECREF(res);
            return true;
        }
    };

    template <typename TObjType>
    class TBaseAttrSetter {
    public:
        virtual ~TBaseAttrSetter() {
        }

        virtual bool SetAttr(PyObject* owner, TObjType& self, const TString& attr, PyObject* val) = 0;
    };

    template <typename TObjType>
    class TAttrGetters {
    public:
        typedef TSimpleSharedPtr<TBaseAttrGetter<TObjType>> TGetterPtr;

    private:
        typedef TVector<TGetterPtr> TGetterList;
        typedef TMap<TString, TGetterList> TGetterMap;

        const TSet<TString>& HiddenAttrNames;
        TGetterMap Getters;

    public:
        TAttrGetters(const TSet<TString>& hiddenNames)
            : HiddenAttrNames(hiddenNames)
        {
        }

        void AddGetter(const TString& attr, TGetterPtr getter) {
            Getters[attr].push_back(getter);
        }

        PyObject* GetAttr(PyObject* owner, const TObjType& self, const TString& attr) const {
            typename TGetterMap::const_iterator it1 = Getters.find(attr);
            if (it1 == Getters.end())
                it1 = Getters.find("");
            if (it1 == Getters.end())
                return nullptr;
            const TGetterList& lst = it1->second;
            for (typename TGetterList::const_iterator it2 = lst.begin(), end = lst.end(); it2 != end; ++it2) {
                PyObject* res = nullptr;
                if ((*it2)->GetAttr(owner, self, attr, res))
                    return res;
                // IMPORTANT!
                // we have to fail GetAttr right there  because we've failed because of internal python error/exception and can't continue iterating because
                // it cause subsequent exceptions during call to Py_BuildValue
                // moreover we have to preserve original exception right there
                if (PyErr_Occurred()) {
                    break;
                }
            }
            return nullptr;
        }

        bool HasAttr(PyObject* owner, const TObjType& self, const TString& attr) const {
            typename TGetterMap::const_iterator it1 = Getters.find(attr);
            if (it1 == Getters.end())
                return false;
            const TGetterList& lst = it1->second;
            for (typename TGetterList::const_iterator it2 = lst.begin(), end = lst.end(); it2 != end; ++it2) {
                if ((*it2)->HasAttr(owner, self, attr, HiddenAttrNames))
                    return true;
            }
            return false;
        }

        void GetAttrsDictionary(PyObject* owner, const TObjType& self, TMap<TString, PyObject*>& res) const {
            for (typename TGetterMap::const_iterator it = Getters.begin(), end = Getters.end(); it != end; ++it) {
                try {
                    if (HasAttr(owner, self, it->first)) {
                        auto attrPtr = GetAttr(owner, self, it->first);
                        if (attrPtr) {
                            res[it->first] = attrPtr;
                        }
                        if (PyErr_Occurred()) {
                            PyErr_Clear(); // Skip python errors as well
                        }
                    }
                } catch (const std::exception&) {
                    // ignore this field
                }
            }
        }

        void GetAttrsNames(PyObject* owner, const TObjType& self, TVector<TString>& resultNames) const {
            for (typename TGetterMap::const_iterator it = Getters.begin(), end = Getters.end(); it != end; ++it) {
                if (HasAttr(owner, self, it->first))
                    resultNames.push_back(it->first);
            }
        }
    };

    template <typename TObjType>
    class TGenericAttrGetter: public TBaseAttrGetter<TObjType> {
    private:
        TString AttrName;

    public:
        TGenericAttrGetter(const TString& attrName)
            : AttrName(attrName)
        {
        }

        bool GetAttr(PyObject* obj, const TObjType&, const TString&, PyObject*& res) const override {
            auto str = NameFromString(AttrName);
            res = PyObject_GenericGetAttr(obj, str.Get());
            if (!res && !PyErr_Occurred())
                ythrow TPyErr(PyExc_AttributeError) << "Can't get generic attribute '" << AttrName << "'";
            return res;
        }
    };

    template <typename TObjType>
    class TAttrSetters {
    private:
        typedef TSimpleSharedPtr<TBaseAttrSetter<TObjType>> TSetterPtr;
        typedef TVector<TSetterPtr> TSetterList;
        typedef TMap<TString, TSetterList> TSetterMap;

        TSetterMap Setters;

    public:
        void AddSetter(const TString& attr, TSetterPtr setter) {
            Setters[attr].push_back(setter);
        }

        bool SetAttr(PyObject* owner, TObjType& self, const TString& attr, PyObject* val) {
            typename TSetterMap::const_iterator it1 = Setters.find(attr);
            if (it1 == Setters.end())
                it1 = Setters.find("");
            if (it1 == Setters.end())
                return false;
            const TSetterList& lst = it1->second;
            for (typename TSetterList::const_iterator it2 = lst.begin(), end = lst.end(); it2 != end; ++it2) {
                if ((*it2)->SetAttr(owner, self, attr, val))
                    return true;
            }
            return false;
        }

        bool SetAttrDictionary(PyObject* owner, TObjType& self, TMap<TString, PyObject*>& dict) {
            for (TMap<TString, PyObject*>::const_iterator it = dict.begin(), end = dict.end(); it != end; ++it) {
                try {
                    SetAttr(owner, self, it->first, it->second);
                } catch (std::exception&) {
                    // ignore this field
                }
            }

            return true;
        }
    };

    /**
      * TMethodAttrGetter - this class maps Python attribute read to C++ method call
      */
    template <typename TObjType, typename TResult, typename TSubObject>
    class TMethodAttrGetter: public TBaseAttrGetter<TObjType> {
    private:
        typedef TResult (TSubObject::*TMethod)() const;
        TMethod Method;

    public:
        TMethodAttrGetter(TMethod method)
            : Method(method)
        {
        }

        bool GetAttr(PyObject*, const TObjType& self, const TString&, PyObject*& res) const override {
            const TSubObject* sub = dynamic_cast<const TSubObject*>(&self);
            if (sub == nullptr)
                return false;
            res = BuildPyObject((sub->*Method)());
            return (res != nullptr);
        }
    };

    template <typename TObjType, typename TFunctor>
    class TFunctorAttrGetter: public TBaseAttrGetter<TObjType> {
        TFunctor Functor;
    public:
        explicit TFunctorAttrGetter(TFunctor functor)
            : Functor(functor)
        {
        }

        bool GetAttr(PyObject*, const TObjType& self, const TString&, PyObject*& res) const override {
            res = BuildPyObject(Functor(self));
            return (res != nullptr);
        }
    };


    /**
      * TMethodAttrGetterWithCheck - this class maps Python attribute read to C++ HasAttr/GetAttr call
      *     If HasAttr returns false, None is returned.
      *     Otherwise GetAttr is called.
      */
    template <typename TObjType, typename TResult, typename TSubObject>
    class TMethodAttrGetterWithCheck: public TBaseAttrGetter<TObjType> {
    private:
        typedef TResult (TSubObject::*TMethod)() const;
        typedef bool (TSubObject::*TCheckerMethod)() const;
        TMethod Method;
        TCheckerMethod CheckerMethod;

    public:
        TMethodAttrGetterWithCheck(TMethod method, TCheckerMethod checkerMethod)
            : Method(method)
            , CheckerMethod(checkerMethod)
        {
        }

        bool GetAttr(PyObject*, const TObjType& self, const TString&, PyObject*& res) const override {
            const TSubObject* sub = dynamic_cast<const TSubObject*>(&self);
            if (sub == nullptr)
                return false;
            if ((sub->*CheckerMethod)())
                res = BuildPyObject((sub->*Method)());
            else {
                Py_INCREF(Py_None);
                res = Py_None;
            }
            return (res != nullptr);
        }
    };

    template <typename TObjType, typename TResult, typename TSubObject, typename TMapper>
    class TMethodAttrMappingGetter: public TBaseAttrGetter<TObjType> {
    private:
        typedef TResult (TSubObject::*TMethod)() const;

        TMethod Method;
        TMapper Mapper;

    public:
        TMethodAttrMappingGetter(TMethod method, TMapper mapper)
            : Method(method)
            , Mapper(mapper)
        {
        }

        bool GetAttr(PyObject*, const TObjType& self, const TString&, PyObject*& res) const override {
            const TSubObject* sub = dynamic_cast<const TSubObject*>(&self);
            if (sub == nullptr)
                return false;
            res = BuildPyObject(Mapper((sub->*Method)()));
            return (res != nullptr);
        }
    };

    template <typename TObjType, typename TResult, typename TSubObject, typename TMapper>
    TSimpleSharedPtr<TBaseAttrGetter<TObjType>>
    CreateMethodAttrMappingGetter(TResult (TSubObject::*method)() const,
                                  TMapper mapper) {
        return new TMethodAttrMappingGetter<TObjType, TResult, TSubObject, TMapper>(method,
                                                                                    mapper);
    }

    template <typename TObjType, typename TResult, typename TValue, typename TSubObject>
    class TMethodAttrSetter: public TBaseAttrSetter<TObjType> {
    private:
        typedef TResult (TSubObject::*TMethod)(TValue&);
        TMethod Method;

    public:
        TMethodAttrSetter(TMethod method)
            : Method(method)
        {
        }

        virtual bool SetAttr(PyObject*, TObjType& self, const TString&, PyObject* val) {
            TSubObject* sub = dynamic_cast<TSubObject*>(&self);
            if (sub == nullptr)
                return false;
            TValue value;
            if (!FromPyObject(val, value))
                return false;
            (sub->*Method)(value);
            return true;
        }
    };

    template <typename TObjType, typename TValue, typename TFunctor>
    class TFunctorAttrSetter: public TBaseAttrSetter<TObjType> {
        TFunctor Functor;
    public:
        explicit TFunctorAttrSetter(TFunctor functor)
            : Functor(functor)
        {
        }

        bool SetAttr(PyObject*, TObjType& self, const TString&, PyObject* val) const override {
            TValue value;
            if (!FromPyObject(val, value))
                return false;
            auto res = BuildPyObject(Functor(self, value));
            return (res != nullptr);
        }
    };
    template <typename TObjType, typename TResult, typename TSubObject>
    TSimpleSharedPtr<TBaseAttrGetter<TObjType>> CreateMethodAttrGetter(TResult (TSubObject::*method)() const) {
        return new TMethodAttrGetter<TObjType, TResult, TSubObject>(method);
    }

    template <typename TObjType, typename TFunctor>
    TSimpleSharedPtr<TFunctorAttrGetter<TObjType, TFunctor>> CreateFunctorAttrGetter(TFunctor functor) {
        return MakeSimpleShared<TFunctorAttrGetter<TObjType, TFunctor>>(functor);
    }

    template <typename TObjType, typename TResult, typename TSubObject>
    TSimpleSharedPtr<TBaseAttrGetter<TObjType>> CreateMethodAttrGetterWithCheck(
        TResult (TSubObject::*method)() const,
        bool (TSubObject::*checkerMethod)() const) {
        return new TMethodAttrGetterWithCheck<TObjType, TResult, TSubObject>(method, checkerMethod);
    }

    template <typename TObjType, typename TResult, typename TValue, typename TSubObject>
    TSimpleSharedPtr<TBaseAttrSetter<TObjType>> CreateMethodAttrSetter(TResult (TSubObject::*method)(TValue&)) {
        return new TMethodAttrSetter<TObjType, TResult, TValue, TSubObject>(method);
    }

    template <typename TObjType, typename TFunctor, typename TValue>
    TSimpleSharedPtr<TFunctorAttrSetter<TObjType, TValue, TFunctor>> CreateFunctorAttrSetter(TFunctor functor) {
        return MakeSimpleShared<TFunctorAttrSetter<TObjType, TValue, TFunctor>>(functor);
    }

    template <typename TObjType, typename TValue, typename TSubObject>
    class TDirectAttrSetter: public TBaseAttrSetter<TObjType> {
    private:
        typedef TValue TSubObject::*TValueType;
        TValueType Value;

    public:
        TDirectAttrSetter(TValueType value)
            : Value(value)
        {
        }

        bool SetAttr(PyObject*, TObjType& self, const TString&, PyObject* val) override {
            TSubObject* sub = dynamic_cast<TSubObject*>(&self);
            if (sub == NULL)
                return false;
            if (!FromPyObject(val, sub->*Value))
                return false;
            return true;
        }
    };

    template <typename TObjType, typename TValue, typename TSubObject>
    TSimpleSharedPtr<TBaseAttrSetter<TObjType>> CreateAttrSetter(TValue TSubObject::*value) {
        return new TDirectAttrSetter<TObjType, TValue, TSubObject>(value);
    }

    template <typename TObjType, typename TValue, typename TSubObject>
    class TDirectAttrGetter: public TBaseAttrGetter<TObjType> {
    private:
        typedef TValue TSubObject::*TValueType;
        TValueType Value;

    public:
        TDirectAttrGetter(TValueType value)
            : Value(value)
        {
        }

        bool GetAttr(PyObject*, const TObjType& self, const TString&, PyObject*& res) const override {
            const TSubObject* sub = dynamic_cast<const TSubObject*>(&self);
            if (sub == nullptr)
                return false;
            res = BuildPyObject(sub->*Value);
            return (res != nullptr);
        }
    };

    template <typename TObjType, typename TValue, typename TSubObject>
    TSimpleSharedPtr<TBaseAttrGetter<TObjType>> CreateAttrGetter(TValue TSubObject::*value) {
        return new TDirectAttrGetter<TObjType, TValue, TSubObject>(value);
    }
}
