#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/generic/set.h>
#include <util/generic/yexception.h>
#include <util/generic/hash.h>
#include <util/generic/map.h>
#include <util/generic/maybe.h>
#include <utility>
#include <initializer_list>
#include "ptr.h"

namespace NPyBind {
    PyObject* GetTrueRef(bool incref = true);
    PyObject* GetFalseRef(bool incref = true);

    PyObject* BuildPyObject(int val);
    PyObject* BuildPyObject(unsigned int val);
    PyObject* BuildPyObject(long int val);
    PyObject* BuildPyObject(unsigned long int val);
#ifdef PY_LONG_LONG
    PyObject* BuildPyObject(PY_LONG_LONG val);
    PyObject* BuildPyObject(unsigned PY_LONG_LONG val);
#endif
    PyObject* BuildPyObject(float val);
    PyObject* BuildPyObject(double val);
    PyObject* BuildPyObject(const TStringBuf& val);
    PyObject* BuildPyObject(const char* val);
    PyObject* BuildPyObject(const TWtringBuf& val);
    PyObject* BuildPyObject(const TBuffer& val);
    PyObject* BuildPyObject(bool val);
    PyObject* BuildPyObject(PyObject*);
    PyObject* BuildPyObject(TPyObjectPtr);

    template <typename T>
    PyObject* BuildPyObject(const TVector<T>& val);

    template <typename T>
    PyObject* BuildPyObject(const TSet<T>& val);

    template <typename TKey, typename TVal>
    PyObject* BuildPyObject(const THashMap<TKey, TVal>& val);

    template <typename T1, typename T2>
    PyObject* BuildPyObject(const std::pair<T1, T2>& val) {
        TPyObjectPtr first(BuildPyObject(val.first), true);
        if (!first) {
            return nullptr;
        }
        TPyObjectPtr second(BuildPyObject(val.second), true);
        if (!first || !second) {
            return nullptr;
        }
        TPyObjectPtr res(PyList_New(2), true);
        PyList_SetItem(res.Get(), 0, first.RefGet());
        PyList_SetItem(res.Get(), 1, second.RefGet());
        return res.RefGet();
    }

    template <typename T>
    PyObject* BuildPyObject(const TVector<T>& val) {
        TPyObjectPtr res(PyList_New(val.size()), true);
        for (size_t i = 0, size = val.size(); i < size; ++i) {
            auto pythonVal = BuildPyObject(std::move(val[i]));
            if (!pythonVal) {
                return nullptr;
            }
            PyList_SetItem(res.Get(), i, pythonVal);
        }
        return res.RefGet();
    }

    template <typename T>
    PyObject* BuildPyObject(TVector<T>&& val) {
        TPyObjectPtr res(PyList_New(val.size()), true);
        for (size_t i = 0, size = val.size(); i < size; ++i) {
            auto pythonVal = BuildPyObject(std::move(val[i]));
            if (!pythonVal) {
                return nullptr;
            }
            PyList_SetItem(res.Get(), i, pythonVal);
        }
        return res.RefGet();
    }

    template <typename T>
    PyObject* BuildPyObject(const TSet<T>& val) {
        TPyObjectPtr res(PySet_New(nullptr), true);
        for (const auto& v : val) {
            auto pythonVal = BuildPyObject(std::move(v));
            if (!pythonVal) {
                return nullptr;
            }
            PySet_Add(res.Get(), pythonVal);
        }
        return res.RefGet();
    }

    template <typename T>
    PyObject* BuildPyObject(const THashSet<T>& val) {
        TPyObjectPtr res(PySet_New(nullptr), true);
        for (const auto& v : val) {
            auto pythonVal = BuildPyObject(std::move(v));
            if (!pythonVal) {
                return nullptr;
            }
            PySet_Add(res.Get(), pythonVal);
        }
        return res.RefGet();
    }

    template <typename TKey, typename TVal>
    PyObject* BuildPyObject(const THashMap<TKey, TVal>& val) {
        TPyObjectPtr res(PyDict_New(), true);
        for (typename THashMap<TKey, TVal>::const_iterator it = val.begin(), end = val.end(); it != end; ++it) {
            auto prevOccurred = PyErr_Occurred();
            Y_UNUSED(prevOccurred);
            TPyObjectPtr k(BuildPyObject(it->first), true);
            if (!k) {
                return nullptr;
            }
            TPyObjectPtr v(BuildPyObject(it->second), true);
            if (!v) {
                return nullptr;
            }
            PyDict_SetItem(res.Get(), k.Get(), v.Get());
        }
        return res.RefGet();
    }

    template <typename TKey, typename TVal>
    PyObject* BuildPyObject(const TMap<TKey, TVal>& val) {
        TPyObjectPtr res(PyDict_New(), true);
        for (typename TMap<TKey, TVal>::const_iterator it = val.begin(), end = val.end(); it != end; ++it) {
            TPyObjectPtr k(BuildPyObject(it->first), true);
            if (!k) {
                return nullptr;
            }
            TPyObjectPtr v(BuildPyObject(it->second), true);
            if (!v) {
                return nullptr;
            }
            PyDict_SetItem(res.Get(), k.Get(), v.Get());
        }
        return res.RefGet();
    }


    template <typename TKey, typename TVal>
    PyObject* BuildPyObject(const TMultiMap<TKey, TVal>& val) {
        TPyObjectPtr res(PyDict_New(), true);
        TMaybe<TKey> prevKey;
        TPyObjectPtr currentEntry(PyList_New(0), true);
        for (const auto& [key, value]: val) {
            if (prevKey.Defined() && prevKey != key) {
                TPyObjectPtr pyPrevKey(BuildPyObject(*prevKey), true);
                if (!pyPrevKey) {
                    return nullptr;
                }
                PyDict_SetItem(res.Get(), pyPrevKey.Get(), currentEntry.Get());
                currentEntry = TPyObjectPtr(PyList_New(0), true);
            }
            TPyObjectPtr pyValue(BuildPyObject(value), true);
            if (!pyValue) {
                return nullptr;
            }
            PyList_Append(currentEntry.Get(), pyValue.Get());
            prevKey = key;
        }

        if (prevKey.Defined()) {
            TPyObjectPtr pyPrevKey(BuildPyObject(*prevKey), true);
            if (!pyPrevKey) {
                return nullptr;
            }
            PyDict_SetItem(res.Get(), pyPrevKey.Get(), currentEntry.Get());
        }
        return res.RefGet();
    }

    template <typename T>
    PyObject* BuildPyObject(const TMaybe<T>& val) {
        if (!val.Defined())
            Py_RETURN_NONE;
        return BuildPyObject(val.GetRef());
    }

    template <typename T, typename C, typename D>
    PyObject* BuildPyObject(const TSharedPtr<T, C, D>& val) {
        if (!val.Get())
            Py_RETURN_NONE;
        return BuildPyObject(*val.Get());
    }

    template <typename T>
    bool FromPyObject(PyObject* obj, T& res);

    bool FromPyObject(PyObject* obj, TString& res);
    bool FromPyObject(PyObject* obj, TStringBuf& res);
    bool FromPyObject(PyObject* obj, TUtf16String& res);
    bool FromPyObject(PyObject* obj, TBuffer& res);

    template <typename T>
    bool FromPyObject(PyObject* obj, TMaybe<T>& res) {
        //we need to save current error before trying derserialize the value
        //because it can produce conversion errors in python that we don't need to handle
        struct TError {
        public:
            TError() {
                PyErr_Fetch(&Type, &Value, &Traceback);
            }
            ~TError() {
                PyErr_Restore(Type, Value, Traceback);

            }
        private:
            PyObject* Type = nullptr;
            PyObject* Value = nullptr;
            PyObject* Traceback = nullptr;
        } currentPyExcInfo;
        T val;
        if (FromPyObject(obj, val)) {
            res = val;
            return true;
        }
        if (obj == Py_None) {
            res = Nothing();
            return true;
        }
        return false;
    }

    template <typename T1, typename T2>
    bool FromPyObject(PyObject* obj, std::pair<T1, T2>& res) {
        PyObject* first;
        PyObject* second;
        if (PyTuple_Check(obj) && 2 == PyTuple_Size(obj)) {
            first = PyTuple_GET_ITEM(obj, 0);
            second = PyTuple_GET_ITEM(obj, 1);
        } else if (PyList_Check(obj) && 2 == PyList_Size(obj)) {
            first = PyList_GET_ITEM(obj, 0);
            second = PyList_GET_ITEM(obj, 1);
        } else {
            return false;
        }
        return FromPyObject(first, res.first) && FromPyObject(second, res.second);
    }

    template <typename T>
    bool FromPyObject(PyObject* obj, TVector<T>& res) {
        if (!PyList_Check(obj))
            return false;
        size_t cnt = PyList_Size(obj);
        res.resize(cnt);
        for (size_t i = 0; i < cnt; ++i) {
            PyObject* item = PyList_GET_ITEM(obj, i);
            if (!FromPyObject(item, res[i]))
                return false;
        }
        return true;
    }

    template <typename K, typename V>
    bool FromPyObject(PyObject* obj, THashMap<K, V>& res) {
        if (!PyDict_Check(obj))
            return false;
        TPyObjectPtr list(PyDict_Keys(obj), true);
        size_t cnt = PyList_Size(list.Get());
        for (size_t i = 0; i < cnt; ++i) {
            PyObject* key = PyList_GET_ITEM(list.Get(), i);
            PyObject* value = PyDict_GetItem(obj, key);
            K rkey;
            V rvalue;
            if (!FromPyObject(key, rkey))
                return false;
            if (!FromPyObject(value, rvalue))
                return false;
            res[rkey] = rvalue;
        }
        return true;
    }

    template <typename K, typename V>
    bool FromPyObject(PyObject* obj, TMap<K, V>& res) {
        if (!PyDict_Check(obj))
            return false;
        TPyObjectPtr list(PyDict_Keys(obj), true);
        size_t cnt = PyList_Size(list.Get());
        for (size_t i = 0; i < cnt; ++i) {
            PyObject* key = PyList_GET_ITEM(list.Get(), i);
            PyObject* value = PyDict_GetItem(obj, key);
            K rkey;
            V rvalue;
            if (!FromPyObject(key, rkey))
                return false;
            if (!FromPyObject(value, rvalue))
                return false;
            res[rkey] = rvalue;
        }
        return true;
    }

    class cast_exception: public TBadCastException {
    };

    template <typename T>
    T FromPyObject(PyObject* obj) {
        T res;
        if (!FromPyObject(obj, res))
            ythrow cast_exception() << "Cannot cast argument to " << TypeName<T>();
        return res;
    }

    template <class... Args, std::size_t... I>
    bool ExtractArgs(std::index_sequence<I...>, PyObject* args, Args&... outArgs) {
        if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != sizeof...(Args))
            return false;
        bool res = true;
        (void)std::initializer_list<bool>{(res = res && NPyBind::FromPyObject(PyTuple_GET_ITEM(args, I), outArgs))...};
        return res;
    }

    template <class... Args>
    bool ExtractArgs(PyObject* args, Args&... outArgs) {
        return ExtractArgs(std::index_sequence_for<Args...>(), args, outArgs...);
    }

    template <class... Args, std::size_t... I>
    bool ExtractOptionalArgs(std::index_sequence<I...>, PyObject* args, PyObject* kwargs, const char* keywords[], Args&... outArgs) {
        PyObject* pargs[sizeof...(Args)] = {};
        static const char format[sizeof...(Args) + 2] = {'|', ((void)I, 'O')..., 0};
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, const_cast<char**>(keywords), &pargs[I]...))
            return false;
        bool res = true;
        (void)std::initializer_list<bool>{(res = res && (!pargs[I] || NPyBind::FromPyObject(pargs[I], outArgs)))...};
        return res;
    }

    template <class... Args>
    bool ExtractOptionalArgs(PyObject* args, PyObject* kwargs, const char* keywords[], Args&... outArgs) {
        return ExtractOptionalArgs(std::index_sequence_for<Args...>(), args, kwargs, keywords, outArgs...);
    }

    template <typename... Args, std::size_t... I>
    static auto GetArguments(std::index_sequence<I...>, PyObject* args) {
        Y_UNUSED(args); // gcc bug
        return std::make_tuple(FromPyObject<std::remove_cv_t<std::remove_reference_t<Args>>>(PyTuple_GetItem(args, I))...);
    }

    template <typename... Args>
    static auto GetArguments(PyObject* args) {
        return GetArguments<Args...>(std::index_sequence_for<Args...>(), args);
    }

    inline PyObject* ReturnString(TStringBuf s) {
#if PY_MAJOR_VERSION >= 3
        return PyUnicode_FromStringAndSize(s.data(), s.size());
#else
        return PyBytes_FromStringAndSize(s.data(), s.size());
#endif
    }

    inline TPyObjectPtr ReturnBytes(TStringBuf s) {
        return TPyObjectPtr(PyBytes_FromStringAndSize(s.data(), s.size()), true);
    }

    inline TPyObjectPtr NameFromString(TStringBuf s) {
        return TPyObjectPtr(ReturnString(s), true);
    }
}
