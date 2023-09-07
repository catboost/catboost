#include "loads.h"

#include <Python.h>

#include <library/cpp/json/fast_sax/parser.h>

#include <util/generic/algorithm.h>
#include <util/generic/stack.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <util/string/ascii.h>

using namespace NJson;

namespace {
    enum EKind {
        Undefined,
        Array,
        Dict,
        Value,
        Key,
    };

    static inline TStringBuf ToStr(EKind kind) noexcept {
        switch (kind) {
            case Undefined:
                return TStringBuf("Undefined");

            case Array:
                return TStringBuf("Array");

            case Dict:
                return TStringBuf("Dict");

            case Value:
                return TStringBuf("Value");

            case Key:
                return TStringBuf("Key");
        }

        Y_UNREACHABLE();
    }

    struct TUnref {
        static inline void Destroy(PyObject* o) noexcept {
            Py_XDECREF(o);
        }
    };

    using TObjectPtr = TAutoPtr<PyObject, TUnref>;

    static inline TObjectPtr BuildBool(bool val) noexcept {
        if (val) {
            Py_RETURN_TRUE;
        }

        Py_RETURN_FALSE;
    }

    // Translate python exceptions from object-creating functions into c++ exceptions
    // Such errors are reported by returning nullptr
    // When a python error is set and C++ exception is caught by Cython wrapper,
    // Python exception is propagated, while C++ exception is discarded.
    PyObject* CheckNewObject(PyObject* obj) {
        Y_ENSURE(obj != nullptr, "got python exception");
        return obj;
    }

    void CheckRetcode(int retcode) {
        Y_ENSURE(retcode == 0, "got python exception");
    }

    static inline TObjectPtr BuildSmall(long val) {
#if PY_VERSION_HEX >= 0x03000000
        return CheckNewObject(PyLong_FromLong(val));
#else
        return CheckNewObject(PyInt_FromLong(val));
#endif
    }

    PyObject* CreatePyString(TStringBuf str, bool intern, bool mayUnicode) {
#if PY_VERSION_HEX >= 0x03000000
        Y_UNUSED(mayUnicode);
        PyObject* pyStr = PyUnicode_FromStringAndSize(str.data(), str.size());
        if (intern) {
            PyUnicode_InternInPlace(&pyStr);
        }
#else
        const bool needUnicode = mayUnicode && !AllOf(str, IsAscii);
        PyObject* pyStr = needUnicode ? PyUnicode_FromStringAndSize(str.data(), str.size())
                                      : PyString_FromStringAndSize(str.data(), str.size());
        if (intern && !needUnicode) {
            PyString_InternInPlace(&pyStr);
        }
#endif
        return pyStr;
    }

    struct TVal {
        EKind Kind = Undefined;
        TObjectPtr Val;

        inline TVal() noexcept
            : Kind(Undefined)
        {
        }

        inline TVal(EKind kind, TObjectPtr val) noexcept
            : Kind(kind)
            , Val(val)
        {
        }
    };

    static inline TObjectPtr NoneRef() noexcept {
        Py_RETURN_NONE;
    }

    struct TContext: public TJsonCallbacks {
        const bool InternKeys;
        const bool InternVals;
        const bool MayUnicode;
        TStack<TVal, TVector<TVal>> S;

        inline TContext(bool internKeys, bool internVals, bool mayUnicode)
            : TJsonCallbacks(true)
            , InternKeys(internKeys)
            , InternVals(internVals)
            , MayUnicode(mayUnicode)
        {
            S.emplace();
        }

        inline bool Consume(TObjectPtr o) {
            auto& t = S.top();

            if (t.Kind == Array) {
                CheckRetcode(PyList_Append(t.Val.Get(), o.Get()));
            } else if (t.Kind == Key) {
                auto key = S.top().Val;

                S.pop();

                CheckRetcode(PyDict_SetItem(S.top().Val.Get(), key.Get(), o.Get()));
            } else {
                t = TVal(Value, o);
            }

            return true;
        }

        inline TObjectPtr Pop(EKind expect) {
            auto res = S.top();

            S.pop();

            if (res.Kind != expect) {
                ythrow yexception() << "unexpected kind(expect " << ToStr(expect) << ", got " << ToStr(res.Kind) << ")";
            }

            return res.Val;
        }

        inline void Push(EKind kind, TObjectPtr object) {
            S.push(TVal(kind, object));
        }

        virtual bool OnNull() {
            return Consume(NoneRef());
        }

        virtual bool OnBoolean(bool v) {
            return Consume(BuildBool(v));
        }

        virtual bool OnInteger(long long v) {
            if (v >= (long long)Min<long>()) {
                return Consume(BuildSmall((long)v));
            }

            return Consume(CheckNewObject(PyLong_FromLongLong(v)));
        }

        virtual bool OnUInteger(unsigned long long v) {
            if (v <= (unsigned long long)Max<long>()) {
                return Consume(BuildSmall((long)v));
            }

            return Consume(CheckNewObject(PyLong_FromUnsignedLongLong(v)));
        }

        virtual bool OnDouble(double v) {
            return Consume(CheckNewObject(PyFloat_FromDouble(v)));
        }

        virtual bool OnString(const TStringBuf& v) {
            return Consume(CheckNewObject(CreatePyString(v, InternVals, MayUnicode)));
        }

        virtual bool OnOpenMap() {
            Push(Dict, CheckNewObject(PyDict_New()));

            return true;
        }

        virtual bool OnCloseMap() {
            return Consume(Pop(Dict));
        }

        virtual bool OnMapKey(const TStringBuf& k) {
            Push(Key, CheckNewObject(CreatePyString(k, InternKeys, MayUnicode)));
            return true;
        }

        virtual bool OnOpenArray() {
            Push(Array, CheckNewObject(PyList_New(0)));

            return true;
        }

        virtual bool OnCloseArray() {
            return Consume(Pop(Array));
        }
    };
}

PyObject* LoadJsonFromString(const char* data, size_t len, bool internKeys, bool internVals, bool mayUnicode) {
    TContext ctx(internKeys, internVals, mayUnicode);

    if (!len) {
        ythrow yexception() << "parse error: zero length input string";
    }

    if (!NJson::ReadJsonFast(TStringBuf(data, len), &ctx)) {
        ythrow yexception() << "parse error";
    }

    auto& s = ctx.S;

    if (!s || s.top().Kind != Value) {
        ythrow yexception() << "shit happen";
    }

    return s.top().Val.Release();
}
