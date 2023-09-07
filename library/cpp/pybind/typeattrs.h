#pragma once

#include "ptr.h"
#include "cast.h"
#include "attr.h"
#include "method.h"

#include <util/generic/vector.h>

namespace NPyBind {
    template <typename TObject>
    class TPythonTypeAttributes {
    private:
        TAttrGetters<TObject> AttrGetters;
        TAttrSetters<TObject> AttrSetters;
        TMethodCallers<TObject> MethodCallers;

        class TGetAttrsNamesCaller;
        class TGetMethodsNamesCaller;
        class TGetAllNamesCaller;
        class TGetPropertiesNamesCaller;
        class TDictAttrGetter;
        class TDictAttrSetter;
        class TGetAttributeMethodCaller;
        class TSetAttrMethodCaller;
        class TGetStrReprMethodCaller;
        class TReduceMethodCaller;
        class TBaseGetStateMethodCaller;
        class TBaseSetStateMethodCaller;

        TPythonTypeAttributes(const TPythonTypeAttributes&);
        TPythonTypeAttributes& operator=(const TPythonTypeAttributes&);

        static const TSet<TString> HiddenAttrNames;

        typedef PyObject* (*GetAttrFunction)(PyObject*, char*);
        typedef int (*SetAttrFunction)(PyObject*, char*, PyObject*);
        GetAttrFunction GetAttr;
        SetAttrFunction SetAttr;

    public:
        typedef TSimpleSharedPtr<TBaseAttrGetter<TObject>> TGetterPtr;
        typedef TSimpleSharedPtr<TBaseAttrSetter<TObject>> TSetterPtr;
        typedef TSimpleSharedPtr<TBaseMethodCaller<TObject>> TCallerPtr;

        TPythonTypeAttributes(GetAttrFunction getAttr, SetAttrFunction setAttr)
            : AttrGetters(HiddenAttrNames)
            , MethodCallers(HiddenAttrNames)
            , GetAttr(getAttr)
            , SetAttr(setAttr)
        {
        }

        void InitCommonAttributes() {
            // attributes
            AddGetter("__dict__", new TDictAttrGetter(AttrGetters));
            AddSetter("__dict__", new TDictAttrSetter(AttrSetters));

            // methods
            AddCaller("GetAttrsNames", new TGetAttrsNamesCaller(AttrGetters));
            AddCaller("GetMethodsNames", new TGetMethodsNamesCaller(MethodCallers));
            AddCaller("GetAllNames", new TGetAllNamesCaller(AttrGetters, MethodCallers));
            AddCaller("GetPropertiesNames", new TGetPropertiesNamesCaller(MethodCallers));
            AddCaller("__getattribute__", new TGetAttributeMethodCaller(GetAttr));
            AddCaller("__setattr__", new TSetAttrMethodCaller(SetAttr));
            AddCaller("__str__", new TGetStrReprMethodCaller("__str__"));
            AddCaller("__repr__", new TGetStrReprMethodCaller("__repr__"));
            AddCaller("__reduce_ex__", new TReduceMethodCaller);
            AddCaller("__reduce__", new TReduceMethodCaller);
            AddCaller("__getstate__", new TBaseGetStateMethodCaller);
            AddCaller("__setstate__", new TBaseSetStateMethodCaller);

            // generics
            AddGetter("__class__", new TGenericAttrGetter<TObject>("__class__"));
            AddGetter("__doc__", new TGenericAttrGetter<TObject>("__doc__"));
            AddCaller("__sizeof__", new TGenericMethodCaller<TObject>("__sizeof__"));
            AddCaller("__hash__", new TGenericMethodCaller<TObject>("__hash__"));
        }

        void AddGetter(const TString& attr, TGetterPtr getter) {
            AttrGetters.AddGetter(attr, getter);
        }

        void AddSetter(const TString& attr, TSetterPtr setter) {
            AttrSetters.AddSetter(attr, setter);
        }

        void AddCaller(const TString& name, TCallerPtr caller) {
            MethodCallers.AddCaller(name, caller);
        }

        const TAttrGetters<TObject>& GetAttrGetters() const {
            return AttrGetters;
        }

        TAttrSetters<TObject>& GetAttrSetters() {
            return AttrSetters;
        }

        const TMethodCallers<TObject>& GetMethodCallers() const {
            return MethodCallers;
        }

        const TSet<TString>& GetHiddenAttrs() const {
            return HiddenAttrNames;
        }
    };

    template <typename TObjType>
    class TPythonTypeAttributes<TObjType>::TGetAttrsNamesCaller: public TBaseMethodCaller<TObjType> {
    private:
        const TAttrGetters<TObjType>& AttrGetters;

    public:
        TGetAttrsNamesCaller(const TAttrGetters<TObjType>& getters)
            : AttrGetters(getters)
        {
        }

        bool CallMethod(PyObject* owner, TObjType* self, PyObject* args, PyObject*, PyObject*& res) const override {
            if (!ExtractArgs(args))
                ythrow yexception() << "Could not parse args for GetAttrsNames() - it should be none";
            TVector<TString> names;
            AttrGetters.GetAttrsNames(owner, *self, names);
            res = BuildPyObject(names);
            return (res != nullptr);
        }
    };

    template <typename TObjType>
    class TPythonTypeAttributes<TObjType>::TGetMethodsNamesCaller: public TBaseMethodCaller<TObjType> {
    private:
        const TMethodCallers<TObjType>& MethodCallers;

    public:
        TGetMethodsNamesCaller(const TMethodCallers<TObjType>& callers)
            : MethodCallers(callers)
        {
        }

        bool CallMethod(PyObject* owner, TObjType* self, PyObject* args, PyObject*, PyObject*& res) const override {
            if (!ExtractArgs(args))
                ythrow yexception() << "Could not parse args for GetMethodsNames() - it should be none";
            TVector<TString> names;
            MethodCallers.GetMethodsNames(owner, self, names);
            res = BuildPyObject(names);
            return (res != nullptr);
        }
    };

    template <typename TObjType>
    class TPythonTypeAttributes<TObjType>::TGetAllNamesCaller: public TBaseMethodCaller<TObjType> {
    private:
        const TAttrGetters<TObjType>& AttrGetters;
        const TMethodCallers<TObjType>& MethodCallers;

    public:
        TGetAllNamesCaller(const TAttrGetters<TObjType>& getters,
                           const TMethodCallers<TObjType>& callers)
            : AttrGetters(getters)
            , MethodCallers(callers)
        {
        }

        bool CallMethod(PyObject* owner, TObjType* self, PyObject* args, PyObject*, PyObject*& res) const override {
            if (!ExtractArgs(args))
                ythrow yexception() << "Could not parse args for GetAllNames() - it should be none";
            TVector<TString> names;
            AttrGetters.GetAttrsNames(owner, *self, names);
            MethodCallers.GetMethodsNames(owner, self, names);
            res = BuildPyObject(names);
            return (res != nullptr);
        }
    };

    template <typename TObjType>
    class TPythonTypeAttributes<TObjType>::TGetPropertiesNamesCaller: public TBaseMethodCaller<TObjType> {
    private:
        const TMethodCallers<TObjType>& MethodCallers;

    public:
        TGetPropertiesNamesCaller(const TMethodCallers<TObjType>& callers)
            : MethodCallers(callers)
        {
        }

    public:
        bool CallMethod(PyObject* obj, TObjType* self, PyObject* args, PyObject*, PyObject*& res) const override {
            if (!ExtractArgs(args))
                return false;

            TVector<TString> names;
            MethodCallers.GetPropertiesNames(obj, self, names);
            res = BuildPyObject(names);
            return (res != nullptr);
        }
    };

    template <typename TObjType>
    class TPythonTypeAttributes<TObjType>::TDictAttrGetter: public TBaseAttrGetter<TObjType> {
    private:
        TAttrGetters<TObjType>& AttrGetters;

    public:
        TDictAttrGetter(TAttrGetters<TObjType>& getters)
            : AttrGetters(getters)
        {
        }

        bool GetAttr(PyObject* owner, const TObjType& self, const TString&, PyObject*& res) const override {
            TMap<TString, PyObject*> dict;
            AttrGetters.GetAttrsDictionary(owner, self, dict);
            res = BuildPyObject(dict);
            return (res != nullptr);
        }
    };

    template <typename TObjType>
    class TPythonTypeAttributes<TObjType>::TDictAttrSetter: public TBaseAttrSetter<TObjType> {
    private:
        TAttrSetters<TObjType>& AttrSetters;

    public:
        TDictAttrSetter(TAttrSetters<TObjType>& setters)
            : AttrSetters(setters)
        {
        }

        bool SetAttr(PyObject* owner, TObjType& self, const TString&, PyObject* val) override {
            TMap<TString, PyObject*> dict;
            if (!FromPyObject(val, dict))
                ythrow yexception() << "'__dict__' should be set to dictionary";
            if (!AttrSetters.SetAttrDictionary(owner, self, dict))
                return false;
            return true;
        }
    };

    template <typename TObjType>
    class TPythonTypeAttributes<TObjType>::TGetAttributeMethodCaller: public TBaseMethodCaller<TObjType> {
    private:
        GetAttrFunction GetAttr;

    public:
        TGetAttributeMethodCaller(GetAttrFunction getAttr)
            : GetAttr(getAttr)
        {
        }

        bool CallMethod(PyObject* owner, TObjType*, PyObject* args, PyObject*, PyObject*& res) const override {
            TString attrName;
            if (!ExtractArgs(args, attrName))
                ythrow yexception() << "Could not parse args for '__getattribute__' - it should be one string";
            res = GetAttr(owner, const_cast<char*>(attrName.c_str()));
            if (!res)
                // Error already set
                return false;
            return true;
        }
    };

    template <typename TObjType>
    class TPythonTypeAttributes<TObjType>::TSetAttrMethodCaller: public TBaseMethodCaller<TObjType> {
    private:
        SetAttrFunction SetAttr;

    public:
        TSetAttrMethodCaller(SetAttrFunction setAttr)
            : SetAttr(setAttr)
        {
        }

        bool CallMethod(PyObject* owner, TObjType*, PyObject* args, PyObject*, PyObject*& res) const override {
            TString attrName;
            TPyObjectPtr value;
            if (!ExtractArgs(args, attrName, value))
                ythrow yexception() << "Could not parse args for '__setattr__' - it should be one string and value";
            Py_INCREF(Py_None);
            res = Py_None;
            if (-1 == SetAttr(owner, const_cast<char*>(attrName.c_str()), value.Get()))
                // Error already set
                return false;
            return true;
        }
    };

    template <typename TObjType>
    class TPythonTypeAttributes<TObjType>::TGetStrReprMethodCaller: public TBaseMethodCaller<TObjType> {
    private:
        TString MethodName;

    private:
        const TString GetFullName(PyObject* obj) const {
            TString module, name;
            TPyObjectPtr type(PyObject_Type(obj), true);
            if (!FromPyObject(PyObject_GetAttrString(type.Get(), "__module__"), module) || !FromPyObject(PyObject_GetAttrString(type.Get(), "__name__"), name))
                ythrow yexception() << "Could not get name of object";
            return module + "." + name;
        }

    public:
        TGetStrReprMethodCaller(const TString& methodName)
            : MethodName(methodName)
        {
        }

        bool CallMethod(PyObject* owner, TObjType*, PyObject* args, PyObject*, PyObject*& res) const override {
            if (args && !ExtractArgs(args))
                ythrow yexception() << "Could not parse args for '" << MethodName << "'";
            TString message = TString("<") + GetFullName(owner) + " object>";
            res = ReturnString(message);
            return (res != nullptr);
        }
    };

    template <typename TObjType>
    class TPythonTypeAttributes<TObjType>::TReduceMethodCaller: public TBaseMethodCaller<TObjType> {
    public:
        bool CallMethod(PyObject* owner, TObjType*, PyObject*, PyObject*, PyObject*& res) const override {
            TPyObjectPtr tuple(PyTuple_New(3), true);
            // First component: reconstructor
            TPyObjectPtr pybindName(BuildPyObject("pybind"), true);
            TPyObjectPtr mainModule(PyImport_Import(pybindName.Get()), true);
            TPyObjectPtr recName(BuildPyObject("PyBindObjectReconstructor"), true);
            TPyObjectPtr reconstructor(PyObject_GetAttr(mainModule.Get(), recName.Get()), true);
            // Second component: arguments to rebuild object
            TPyObjectPtr arguments(PyTuple_New(2), true);
            TPyObjectPtr cl(PyObject_GetAttrString(owner, "__class__"), true);
            PyTuple_SET_ITEM(arguments.Get(), 0, cl.RefGet());
            TPyObjectPtr props(PyObject_CallMethod(owner, const_cast<char*>("GetPropertiesNames"), nullptr), true);
            PyTuple_SET_ITEM(arguments.Get(), 1, props.RefGet());
            // Third component: state to fill new object
            TPyObjectPtr state(PyObject_CallMethod(owner, const_cast<char*>("__getstate__"), nullptr), true);

            PyTuple_SET_ITEM(tuple.Get(), 0, reconstructor.RefGet());
            PyTuple_SET_ITEM(tuple.Get(), 1, arguments.RefGet());
            PyTuple_SET_ITEM(tuple.Get(), 2, state.RefGet());
            res = tuple.RefGet();
            return (res != nullptr);
        }
    };

    template <typename TObjType>
    class TPythonTypeAttributes<TObjType>::TBaseGetStateMethodCaller: public TGetStateCaller<TObjType, TObjType> {
    public:
        void GetAttrsDictionary(PyObject* obj, TObjType* self, TMap<TString, TPyObjectPtr>& dict) const override {
            this->GetStandartAttrsDictionary(obj, self, dict);
        }
    };

    template <typename TObjType>
    class TPythonTypeAttributes<TObjType>::TBaseSetStateMethodCaller: public TSetStateCaller<TObjType, TObjType> {
    public:
        void SetAttrsDictionary(PyObject* obj, TObjType* self, TMap<TString, TPyObjectPtr>& dict) const override {
            this->SetStandartAttrsDictionary(obj, self, dict);
        }
    };

    static const char* HiddenAttrStrings[] = {
        "__dict__", "__class__", "__dir__", "__delattr__", "__doc__", "__format__", "__getattribute__", "__hash__",
        "__init__", "__new__", "__reduce__", "__reduce_ex__", "__repr__", "__setattr__", "__sizeof__", "__str__",
        "__subclasshook__", "__getstate__", "__setstate__",
        "GetAttrsNames", "GetMethodsNames", "GetAllNames", "GetPropertiesNames"};

    template <typename T>
    const TSet<TString> TPythonTypeAttributes<T>::HiddenAttrNames(HiddenAttrStrings, std::end(HiddenAttrStrings));

}
