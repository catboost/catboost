#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "typeattrs.h"
#include "exceptions.h"
#include "module.h"

namespace NPyBind {
    void RegisterJSONBridge();

    namespace NPrivate {
        template <typename>
        class TUnboundClosureHolder;
        template <typename>
        class TUnboundClosure;
    }

    // TTraits should be derived from TPythonType
    template <typename TObjectHolder, typename TObject, typename TTraits>
    class TPythonType {
    private:
        TPythonType(const TPythonType&);
        TPythonType& operator=(const TPythonType&);

    private:
        typedef typename TPythonTypeAttributes<TObject>::TGetterPtr TGetterPtr;
        typedef typename TPythonTypeAttributes<TObject>::TSetterPtr TSetterPtr;
        typedef typename TPythonTypeAttributes<TObject>::TCallerPtr TCallerPtr;

        struct TProxy {
            PyObject_HEAD
                TObjectHolder* Holder;
        };

        static PyTypeObject PyType;
        static PyMappingMethods MappingMethods;
        static PyObject* PyTypeObjPtr;
    protected:
        static PyTypeObject* GetPyTypePtr() {
            return &PyType;
        }
    private:

        TPythonTypeAttributes<TObject> Attributes;

        static int InitObject(PyObject* s, PyObject* args, PyObject* kwargs) {
            try {
                TProxy* self = reinterpret_cast<TProxy*>(s);
                auto str = NameFromString("__properties__");
                if (kwargs && PyDict_Check(kwargs) && PyDict_Contains(kwargs, str.Get())) {
                    TPyObjectPtr props(PyDict_GetItem(kwargs, str.Get()));
                    TVector<TString> properties;
                    FromPyObject(props.Get(), properties);
                    self->Holder = TTraits::DoInitPureObject(properties);
                } else {
                    self->Holder = (args || kwargs) ? TTraits::DoInitObject(args, kwargs) : nullptr;
                }
                if (PyErr_Occurred())
                    return -1;
                return 0;
            } catch (const TPyNativeErrorException&) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_RuntimeError, "Some PY error occurred, but it is not set.");
                }
            } catch (const std::exception& ex) {
                PyErr_SetString(TExceptionsHolder::Instance().ToPyException(ex).Get(), ex.what());
            } catch (...) {
                PyErr_SetString(PyExc_RuntimeError, "Unknown error occurred while trying to init object");
            }
            return -1;
        }

        static void DeallocObject(TProxy* self) {
            delete self->Holder;
            Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
        }

        static PyObject* GetObjectAttr(PyObject* pyObj, char* attr);
        static int SetObjectAttr(PyObject* pyObj, char* attr, PyObject* value);
        static PyObject* GetStr(PyObject*);
        static PyObject* GetRepr(PyObject*);
        static PyObject* GetIter(PyObject*);
        static PyObject* GetNext(PyObject*);

        // Fill class __dict__ with functions to make sure methods names will get to dir()
        void FillClassDict() const {
            TVector<TString> names;
            Attributes.GetMethodCallers().GetAllMethodsNames(names);
            for (const auto& name : names) {
                TPyObjectPtr callable = NPrivate::TUnboundClosure<TObject>::Instance().CreatePyObject(new NPrivate::TUnboundClosureHolder<TObject>(&PyType, name));
                PyDict_SetItemString(PyType.tp_dict, name.c_str(), callable.Get());
            }
        }

        void InitCommonAttributes() {
            static bool was = false;
            if (was)
                return;
            was = true;
            Attributes.InitCommonAttributes();
            FillClassDict();
        }

    protected:
        TPythonType(const char* pyTypeName, const char* typeDescr, PyTypeObject* parentType = nullptr,
                    TVector<PyTypeObject*> baseTypes = {})
            : Attributes(GetObjectAttr, SetObjectAttr)
        {
            PyType.tp_name = pyTypeName;
            PyType.tp_doc = typeDescr;
            Py_INCREF(PyTypeObjPtr);
            if (parentType) {
                Py_INCREF(parentType);
                PyType.tp_base = parentType;
            }
            if (!baseTypes.empty()) {
                Py_ssize_t baseCount = baseTypes.size();
                PyObject* tuple = PyTuple_New(baseCount);
                for (Py_ssize_t i = 0; i < baseCount; ++i) {
                    Py_INCREF(baseTypes[i]);
                    PyTuple_SET_ITEM(tuple, i, (PyObject *)baseTypes[i]);
                }
                PyType.tp_bases = tuple;
            }
            PyType_Ready(&PyType);

            TExceptionsHolder::Instance();
            RegisterJSONBridge();

        }

        ~TPythonType() {
        }

        static TObjectHolder* DoInitObject(PyObject*, PyObject*) {
            return nullptr;
        }

        static TObjectHolder* DoInitPureObject(const TVector<TString>&) {
            return nullptr;
        }

        static void SetClosure(PyObject* (*call)(PyObject*, PyObject*, PyObject*)) {
            PyType.tp_call = call;
        }

    public:
        void AddGetter(const TString& attr, TGetterPtr getter) {
            Attributes.AddGetter(attr, getter);
        }

        void AddSetter(const TString& attr, TSetterPtr setter) {
            Attributes.AddSetter(attr, setter);
        }

        void AddCaller(const TString& name, TCallerPtr caller) {
            Attributes.AddCaller(name, caller);
            if (name == "__iter__") {
                PyType.tp_iter = GetIter;
            }
            if (name == "next") {
                PyType.tp_iternext = GetNext;
            }
        }

        void SetIter(getiterfunc tp_iter) {
            PyType.tp_iter = tp_iter;
        }

        void SetIterNext(iternextfunc tp_iternext) {
            PyType.tp_iternext = tp_iternext;
        }

        void SetDestructor(destructor tp_dealloc) {
            PyType.tp_dealloc = tp_dealloc;
        }

        void SetLengthFunction(lenfunc mp_length) {
            PyType.tp_as_mapping->mp_length = mp_length;
        }

        void SetSubscriptFunction(binaryfunc mp_subscript) {
            PyType.tp_as_mapping->mp_subscript = mp_subscript;
        }

        void SetAssSubscriptFunction(objobjargproc mp_ass_subscript) {
            PyType.tp_as_mapping->mp_ass_subscript = mp_ass_subscript;
        }

        typedef TObject TObjectType;

        static TPythonType& Instance() {
            static TTraits Traits;
            Traits.InitCommonAttributes();
            return Traits;
        }

        void Register(PyObject* module, const char* typeName) {
            Py_INCREF(PyTypeObjPtr);
            if (0 != PyModule_AddObject(module, typeName, PyTypeObjPtr))
                ythrow yexception() << "can't register type \"" << typeName << "\"";
        }

        void Register(PyObject* module, const char* objName, TObjectHolder* hld) {
            if (0 != PyModule_AddObject(module, objName, CreatePyObject(hld).RefGet()))
                ythrow yexception() << "can't register object \"" << objName << "\"";
        }

        void Register(TPyObjectPtr module, const TString& typeName) {
            Register(module.Get(), typeName.c_str());
        }

        void Register(TPyObjectPtr module, const TString& objName, TObjectHolder* hld) {
            Register(module.Get(), objName.c_str(), hld);
        }

        static TObjectHolder* CastToObjectHolder(PyObject* obj) {
            // Call Instance() to make sure PyTypeObjPtr is already created at this point
            Instance();
            if (!PyObject_IsInstance(obj, PyTypeObjPtr))
                return nullptr;
            TProxy* prx = reinterpret_cast<TProxy*>(obj);
            return prx ? prx->Holder : nullptr;
        }

        static TObject* CastToObject(PyObject* obj) {
            TObjectHolder* hld = CastToObjectHolder(obj);
            return hld ? TTraits::GetObject(*hld) : nullptr;
        }

        static TPyObjectPtr CreatePyObject(TObjectHolder* hld) {
            TPyObjectPtr r(_PyObject_New(&PyType), true);
            TProxy* prx = reinterpret_cast<TProxy*>(r.Get());
            if (prx)
                prx->Holder = hld;
            return r;
        }
    };

    template <typename TObjectHolder, typename TObject, typename TTraits>
    PyMappingMethods TPythonType<TObjectHolder, TObject, TTraits>::MappingMethods = {nullptr, nullptr, nullptr};

    template <typename TObjectHolder, typename TObject, typename TTraits>
    PyTypeObject TPythonType<TObjectHolder, TObject, TTraits>::PyType = {
        PyVarObject_HEAD_INIT(nullptr, 0) "", sizeof(TProxy), 0, (destructor)&DeallocObject
#if PY_VERSION_HEX < 0x030800b4
        , nullptr, /*tp_print*/
#endif
#if PY_VERSION_HEX >= 0x030800b4
        , 0, /*tp_vectorcall_offset*/
#endif
        &GetObjectAttr, &SetObjectAttr, nullptr, &GetRepr, nullptr, nullptr, &MappingMethods, nullptr, nullptr, &GetStr, nullptr, nullptr, nullptr,
        Py_TPFLAGS_DEFAULT, "", nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, InitObject, PyType_GenericAlloc, PyType_GenericNew, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0
#if PY_MAJOR_VERSION >= 3
        , nullptr
#endif
#if PY_VERSION_HEX >= 0x030800b1
        , nullptr /*tp_vectorcall*/
#endif
#if PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
        , nullptr /*tp_print*/
#endif
#if PY_VERSION_HEX >= 0x030C0000
        , 0 /*tp_watched*/
#endif
    };

    template <typename TObjectHolder, typename TObject, typename TTraits>
    PyObject* TPythonType<TObjectHolder, TObject, TTraits>::PyTypeObjPtr =
        reinterpret_cast<PyObject*>(&TPythonType<TObjectHolder, TObject, TTraits>::PyType);

    namespace NPrivate {
        template <typename TObject>
        class TUnboundClosureHolder {
        private:
            THolder<PyTypeObject> Holder;
            TString Method;

        public:
            TUnboundClosureHolder(PyTypeObject* ptr, const TString& meth)
                : Holder(ptr)
                , Method(meth)
            {
            }

            PyTypeObject* GetObject() const {
                return Holder.Get();
            }

            const TString GetMethod() const {
                return Method;
            }

            PyObject* Call(PyObject* obj, PyObject* args, PyObject*) const {
                TPyObjectPtr callable(PyObject_GetAttrString(obj, Method.c_str()), true);
                if (!callable.Get())
                    ythrow yexception() << "PyBind can't call method '" << Method << "'";
                TPyObjectPtr res(PyObject_CallObject(callable.Get(), args), true);
                if (!res.Get() && !PyErr_Occurred())
                    ythrow yexception() << "PyBind can't call method '" << Method << "'";
                return res.RefGet();
            }
        };

        template <typename TObject>
        class TUnboundClosure: public NPyBind::TPythonType<TUnboundClosureHolder<TObject>, PyTypeObject, TUnboundClosure<TObject>> {
        private:
            typedef class NPyBind::TPythonType<TUnboundClosureHolder<TObject>, PyTypeObject, TUnboundClosure<TObject>> TParent;
            friend class NPyBind::TPythonType<TUnboundClosureHolder<TObject>, PyTypeObject, TUnboundClosure<TObject>>;

            class TReprMethodCaller: public TBaseMethodCaller<PyTypeObject> {
            public:
                bool CallMethod(PyObject* closure, PyTypeObject*, PyObject*, PyObject*, PyObject*& res) const override {
                    TUnboundClosureHolder<TObject>* hld = TParent::CastToObjectHolder(closure);
                    TPyObjectPtr type((PyObject*)hld->GetObject());

                    TString nameStr;
                    TPyObjectPtr name(PyObject_GetAttrString(type.Get(), "__name__"), true);
                    if (!name.Get() || !FromPyObject(name.Get(), nameStr))
                        ythrow yexception() << "Could not get name of object";

                    TString methodName(hld->GetMethod());

                    TString message = "<unbound method " + nameStr + "." + methodName + ">";
                    res = ReturnString(message);
                    return (res != nullptr);
                }
            };

        private:
            TUnboundClosure()
                : TParent("", "")
            {
                TParent::AddCaller("__repr__", new TReprMethodCaller());
                TParent::AddCaller("__str__", new TReprMethodCaller());
                TParent::SetClosure(&Call);
            }

            static PyObject* Call(PyObject* closure, PyObject* args, PyObject* kwargs) {
                try {
                    TUnboundClosureHolder<TObject>* hld = TParent::CastToObjectHolder(closure);
                    if (!hld)
                        ythrow yexception() << "Can't cast object to TypeHolder";

                    size_t size = 0;
                    if (!PyTuple_Check(args) || (size = PyTuple_Size(args)) < 1)
                        ythrow yexception() << "Can't parse first argument: it should be valid object";
                    --size;
                    TPyObjectPtr obj(PyTuple_GetItem(args, 0));
                    TPyObjectPtr newArgs(PyTuple_New(size), true);

                    for (size_t i = 0; i < size; ++i) {
                        TPyObjectPtr item(PyTuple_GetItem(args, i + 1));
                        PyTuple_SetItem(newArgs.Get(), i, item.RefGet());
                    }

                    return hld->Call(obj.Get(), newArgs.Get(), kwargs);
                } catch (const TPyNativeErrorException&) {
                    if (!PyErr_Occurred()) {
                        PyErr_SetString(PyExc_RuntimeError, "Some PY error occurred, but it is not set.");
                    }
                } catch (const std::exception& ex) {
                    PyErr_SetString(::NPyBind::TExceptionsHolder::Instance().ToPyException(ex).Get(), ex.what());
                } catch (...) {
                    PyErr_SetString(PyExc_RuntimeError, "Unknown error occurred while trying to call method");
                }
                return nullptr;
            }

            static PyTypeObject* GetObject(TUnboundClosureHolder<TObject>& obj) {
                return obj.GetObject();
            }
        };

        template <typename TObject>
        class TBoundClosureHolder {
        private:
            TPyObjectPtr Ptr;
            TObject* Object;
            TString Method;
            const TMethodCallers<TObject>& MethodCallers;

        public:
            TBoundClosureHolder(PyObject* ptr, TObject* obj, const TString& meth, const TMethodCallers<TObject>& callers)
                : Ptr(ptr)
                , Object(obj)
                , Method(meth)
                , MethodCallers(callers)
            {
            }

            TPyObjectPtr GetObjectPtr() const {
                return Ptr;
            }

            TObject* GetObject() const {
                return Object;
            }

            const TString GetMethod() const {
                return Method;
            }

            PyObject* Call(PyObject* args, PyObject* kwargs) const {
                PyObject* res = MethodCallers.CallMethod(Ptr.Get(), Object, args, kwargs, Method);
                if (res == nullptr && !PyErr_Occurred())
                    ythrow yexception() << "PyBind can't call method '" << Method << "'";
                return res;
            }
        };

        template <typename TObject>
        class TBoundClosure: public TPythonType<TBoundClosureHolder<TObject>, TObject, TBoundClosure<TObject>> {
        private:
            typedef TPythonType<TBoundClosureHolder<TObject>, TObject, TBoundClosure<TObject>> TMyParent;
            class TReprMethodCaller: public TBaseMethodCaller<TObject> {
            public:
                bool CallMethod(PyObject* closure, TObject*, PyObject*, PyObject*, PyObject*& res) const override {
                    TBoundClosureHolder<TObject>* hld = TMyParent::CastToObjectHolder(closure);
                    TPyObjectPtr obj(hld->GetObjectPtr());
                    TPyObjectPtr type(PyObject_Type(obj.Get()), true);

                    TString reprStr;
                    TPyObjectPtr repr(PyObject_Repr(obj.Get()), true);
                    if (!repr.Get() || !FromPyObject(repr.Get(), reprStr))
                        ythrow yexception() << "Could not get repr of object";

                    TString nameStr;
                    TPyObjectPtr name(PyObject_GetAttrString(type.Get(), "__name__"), true);
                    if (!name.Get() || !FromPyObject(name.Get(), nameStr))
                        ythrow yexception() << "Could not get name of object";

                    TString methodName(hld->GetMethod());

                    TString message = "<bound method " + nameStr + "." + methodName + " of " + reprStr + ">";
                    res = ReturnString(message);
                    return (res != nullptr);
                }
            };

        private:
            static PyObject* Call(PyObject* closure, PyObject* args, PyObject* kwargs) {
                try {
                    TBoundClosureHolder<TObject>* hld = TMyParent::CastToObjectHolder(closure);
                    if (!hld)
                        ythrow yexception() << "Can't cast object to ClosureHolder";

                    return hld->Call(args, kwargs);
                } catch (const TPyNativeErrorException&) {
                    if (!PyErr_Occurred()) {
                        PyErr_SetString(PyExc_RuntimeError, "Some PY error occurred, but it is not set.");
                    }
                } catch (const std::exception& ex) {
                    PyErr_SetString(::NPyBind::TExceptionsHolder::Instance().ToPyException(ex).Get(), ex.what());
                } catch (...) {
                    PyErr_SetString(PyExc_RuntimeError, "Unknown error occurred while trying to call method");
                }
                return nullptr;
            }

        public:
            TBoundClosure()
                : TMyParent("", "")
            {
                TMyParent::AddCaller("__repr__", new TReprMethodCaller());
                TMyParent::AddCaller("__str__", new TReprMethodCaller());
                TMyParent::SetClosure(&Call);
            }

            static TObject* GetObject(const TBoundClosureHolder<TObject>& closure) {
                return closure.GetObject();
            }
        };

    }

    template <typename TObjectHolder, typename TObject, typename TTraits>
    PyObject* TPythonType<TObjectHolder, TObject, TTraits>::GetObjectAttr(PyObject* pyObj, char* attr) {
        try {
            TObject* obj = CastToObject(pyObj);
            PyObject* res = obj ? Instance().Attributes.GetAttrGetters().GetAttr(pyObj, *obj, attr) : nullptr;
            if (res == nullptr && Instance().Attributes.GetMethodCallers().HasMethod(pyObj, obj, attr)) {
                TPyObjectPtr r = NPrivate::TBoundClosure<TObject>::Instance().CreatePyObject(new NPrivate::TBoundClosureHolder<TObject>(pyObj, obj, attr, Instance().Attributes.GetMethodCallers()));
                res = r.RefGet();
            }
            if (res == nullptr && !PyErr_Occurred())
                ythrow TPyErr(PyExc_AttributeError) << "PyBind can't get attribute '" << attr << "'";
            return res;
        } catch (const TPyNativeErrorException&) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, "Some PY error occurred, but it is not set.");
            }
        } catch (const std::exception& ex) {
            PyErr_SetString(TExceptionsHolder::Instance().ToPyException(ex).Get(), ex.what());
        } catch (...) {
            PyErr_SetString(PyExc_RuntimeError, (TString("Unknown error occurred while trying to get attribute '") + attr + "'").c_str());
        }
        return nullptr;
    }

    template <typename TObjectHolder, typename TObject, typename TTraits>
    int TPythonType<TObjectHolder, TObject, TTraits>::SetObjectAttr(PyObject* pyObj, char* attr, PyObject* value) {
        try {
            TObject* obj = CastToObject(pyObj);
            bool res = obj ? Instance().Attributes.GetAttrSetters().SetAttr(pyObj, *obj, attr, value) : false;
            if (!res && !PyErr_Occurred())
                ythrow yexception() << "PyBind can't set attribute '" << attr << "'";
            return res ? 0 : -1;
        } catch (const TPyNativeErrorException&) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, "Some PY error occurred, but it is not set.");
            }
        } catch (const std::exception& ex) {
            PyErr_SetString(TExceptionsHolder::Instance().ToPyException(ex).Get(), ex.what());
        } catch (...) {
            PyErr_SetString(PyExc_RuntimeError, (TString("Unknown error occurred while trying to set attribute '") + attr + "'").c_str());
        }
        return -1;
    }

    template <typename TObjectHolder, typename TObject, typename TTraits>
    PyObject* TPythonType<TObjectHolder, TObject, TTraits>::GetStr(PyObject* obj) {
        try {
            TObject* self = CastToObject(obj);
            return Instance().Attributes.GetMethodCallers().CallMethod(obj, self, nullptr, nullptr, "__str__");
        } catch (const TPyNativeErrorException&) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, "Some PY error occurred, but it is not set.");
            }
        } catch (const std::exception& ex) {
            PyErr_SetString(TExceptionsHolder::Instance().ToPyException(ex).Get(), ex.what());
        } catch (...) {
            PyErr_SetString(PyExc_RuntimeError, (TString("Unknown error occurred while trying to call '__str__'").c_str()));
        }
        return nullptr;
    }

    template <typename TObjectHolder, typename TObject, typename TTraits>
    PyObject* TPythonType<TObjectHolder, TObject, TTraits>::GetIter(PyObject* obj) {
        try {
            TObject* self = CastToObject(obj);
            return Instance().Attributes.GetMethodCallers().CallMethod(obj, self, nullptr, nullptr, "__iter__");
        } catch (const TPyNativeErrorException&) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, "Some PY error occurred, but it is not set.");
            }
        } catch (const std::exception& ex) {
            PyErr_SetString(TExceptionsHolder::Instance().ToPyException(ex).Get(), ex.what());
        } catch (...) {
            PyErr_SetString(PyExc_RuntimeError, (TString("Unknown error occurred while trying to call '__iter__'").c_str()));
        }
        return nullptr;
    }

    template <typename TObjectHolder, typename TObject, typename TTraits>
    PyObject* TPythonType<TObjectHolder, TObject, TTraits>::GetNext(PyObject* obj) {
        try {
            TObject* self = CastToObject(obj);
            return Instance().Attributes.GetMethodCallers().CallMethod(obj, self, nullptr, nullptr, "next");
        } catch (const TPyNativeErrorException&) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, "Some PY error occurred, but it is not set.");
            }
        } catch (const std::exception& ex) {
            PyErr_SetString(TExceptionsHolder::Instance().ToPyException(ex).Get(), ex.what());
        } catch (...) {
            PyErr_SetString(PyExc_RuntimeError, (TString("Unknown error occurred while trying to call 'next'").c_str()));
        }
        return nullptr;
    }

    template <typename TObjectHolder, typename TObject, typename TTraits>
    PyObject* TPythonType<TObjectHolder, TObject, TTraits>::GetRepr(PyObject* obj) {
        try {
            TObject* self = CastToObject(obj);
            return Instance().Attributes.GetMethodCallers().CallMethod(obj, self, nullptr, nullptr, "__repr__");
        } catch (const TPyNativeErrorException&) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_RuntimeError, "Some PY error occurred, but it is not set.");
            }
        } catch (const std::exception& ex) {
            PyErr_SetString(TExceptionsHolder::Instance().ToPyException(ex).Get(), ex.what());
        } catch (...) {
            PyErr_SetString(PyExc_RuntimeError, (TString("Unknown error occurred while trying to call '__repr__'").c_str()));
        }
        return nullptr;
    }
}
