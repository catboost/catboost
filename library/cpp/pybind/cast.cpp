#include "cast.h"
#include <util/generic/yexception.h>
#include <util/generic/buffer.h>

namespace NPyBind {
    PyObject* GetTrueRef(bool incref) {
        if (incref)
            Py_RETURN_TRUE;
        return Py_True;
    }

    PyObject* GetFalseRef(bool incref) {
        if (incref)
            Py_RETURN_FALSE;
        return Py_False;
    }

    PyObject* BuildPyObject(int val) {
        return Py_BuildValue("i", val);
    }

    PyObject* BuildPyObject(unsigned int val) {
        return Py_BuildValue("I", val);
    }

    PyObject* BuildPyObject(long int val) {
        return Py_BuildValue("l", val);
    }

    PyObject* BuildPyObject(unsigned long int val) {
        return Py_BuildValue("k", val);
    }

#ifdef PY_LONG_LONG
    PyObject* BuildPyObject(PY_LONG_LONG val) {
        return Py_BuildValue("L", val);
    }

    PyObject* BuildPyObject(unsigned PY_LONG_LONG val) {
        return Py_BuildValue("K", val);
    }
#endif

    PyObject* BuildPyObject(float val) {
        return Py_BuildValue("f", val);
    }

    PyObject* BuildPyObject(double val) {
        return Py_BuildValue("d", val);
    }

    PyObject* BuildPyObject(const TStringBuf& val) {
        if (!val.IsInited())
            Py_RETURN_NONE;

        PyObject* stringValue = Py_BuildValue("s#", val.data(), static_cast<int>(val.length()));
        if (stringValue != nullptr) {
            return stringValue;
        }
        if (PyErr_ExceptionMatches(PyExc_UnicodeDecodeError)) {
            PyErr_Clear();
        } else {
            return nullptr;
        }
        return Py_BuildValue("y#", val.data(), static_cast<int>(val.length()));
    }

    PyObject* BuildPyObject(const char* val) {
        if (val == nullptr)
            Py_RETURN_NONE;
        PyObject* stringValue = Py_BuildValue("s#", val, static_cast<int>(strlen(val)));
        if (stringValue != nullptr) {
            return stringValue;
        }
        if (PyErr_ExceptionMatches(PyExc_UnicodeDecodeError)) {
            PyErr_Clear();
        } else {
            return nullptr;
        }
        return Py_BuildValue("y#", val, static_cast<int>(strlen(val)));
    }

    PyObject* BuildPyObject(const TWtringBuf& val) {
        if (!val.IsInited())
            Py_RETURN_NONE;
#if PY_VERSION_HEX < 0x03030000
        TPyObjectPtr result(PyUnicode_FromUnicode(nullptr, val.size()), true);
        Py_UNICODE* buf = PyUnicode_AS_UNICODE(result.Get());
        if (buf == nullptr)
            Py_RETURN_NONE;
        for (size_t i = 0; i < val.size(); ++i) {
            buf[i] = static_cast<Py_UNICODE>(val[i]);
        }
#else
        PyObject* unicodeValue = PyUnicode_FromKindAndData(PyUnicode_2BYTE_KIND, val.data(), val.size());
        if (unicodeValue == nullptr)
            Py_RETURN_NONE;
        TPyObjectPtr result(unicodeValue, true);
#endif
        return result.RefGet();
    }

    PyObject* BuildPyObject(const TBuffer& val) {
        TPyObjectPtr res(PyList_New(val.size()), true);
        for (size_t i = 0, size = val.Size(); i < size; ++i)
            PyList_SetItem(res.Get(), i, BuildPyObject(val.Data()[i]));
        return res.RefGet();
    }

    PyObject* BuildPyObject(bool val) {
        if (val)
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;
    }

    PyObject* BuildPyObject(PyObject* val) {
        Py_XINCREF(val);
        return val;
    }

    PyObject* BuildPyObject(TPyObjectPtr ptr) {
        return ptr.RefGet();
    }

    /* python represents (http://docs.python.org/c-api/arg.html#Py_BuildValue)
     * char, uchar, short, ushort, int, long as PyInt
     * uint, ulong as PyInt or PyLong (if exceeds sys.maxint)
     * longlong, ulonglong as PyLong
     */

    template <>
    bool FromPyObject(PyObject* obj, long& res) {
        if (PyLong_Check(obj)) {
            res = PyLong_AsLong(obj);
            return !PyErr_Occurred();
        }
        if (PyFloat_Check(obj)) {
            res = static_cast<long>(PyFloat_AsDouble(obj));
            return !PyErr_Occurred();
        }
#if PY_MAJOR_VERSION < 3
        res = PyInt_AsLong(obj);
#else
        return false;
#endif
        return -1 != res || !PyErr_Occurred();
    }

    template <>
    bool FromPyObject(PyObject* obj, unsigned long& res) {
        if (PyLong_Check(obj)) {
            res = PyLong_AsUnsignedLong(obj);
            return !PyErr_Occurred();
        }
        if (PyFloat_Check(obj)) {
            res = static_cast<unsigned long>(PyFloat_AsDouble(obj));
            return !PyErr_Occurred();
        }
#if PY_MAJOR_VERSION < 3
        res = PyInt_AsLong(obj);
#endif
        return !PyErr_Occurred();
    }

    template <>
    bool FromPyObject(PyObject* obj, int& res) {
        long lres;
        if (!FromPyObject(obj, lres))
            return false;
        res = static_cast<int>(lres);
        return true;
    }

    template <>
    bool FromPyObject(PyObject* obj, unsigned char& res) {
        long lres;
        if (!FromPyObject(obj, lres))
            return false;
        res = static_cast<unsigned char>(lres);
        return true;
    }

    template <>
    bool FromPyObject(PyObject* obj, char& res) {
        long lres;
        if (!FromPyObject(obj, lres))
            return false;
        res = static_cast<char>(lres);
        return true;
    }

    template <>
    bool FromPyObject(PyObject* obj, unsigned int& res) {
        unsigned long lres;
        if (!FromPyObject(obj, lres))
            return false;
        res = static_cast<unsigned int>(lres);
        return true;
    }

#ifdef HAVE_LONG_LONG
    template <>
    bool FromPyObject(PyObject* obj, long long& res) {
        if (PyLong_Check(obj)) {
            res = PyLong_AsLongLong(obj);
            return -1 != res || !PyErr_Occurred();
        }
        long lres;
        if (!FromPyObject(obj, lres))
            return false;
        res = static_cast<long long>(lres);
        return true;
    }

    template <>
    bool FromPyObject(PyObject* obj, unsigned long long& res) {
        if (PyLong_Check(obj)) {
            res = PyLong_AsUnsignedLongLong(obj);
            return static_cast<unsigned long long>(-1) != res || !PyErr_Occurred();
        }
        long lres;
        if (!FromPyObject(obj, lres))
            return false;
        res = static_cast<unsigned long long>(lres);
        return true;
    }
#endif

    template <>
    bool FromPyObject(PyObject* obj, double& res) {
        if (PyFloat_Check(obj)) {
            res = PyFloat_AsDouble(obj);
            return true;
        }
        long long lres;
        if (!FromPyObject(obj, lres))
            return false;
        res = static_cast<double>(lres);
        return true;
    }

    template <>
    bool FromPyObject(PyObject* obj, float& res) {
        double dres;
        if (!FromPyObject(obj, dres))
            return false;
        res = static_cast<float>(dres);
        return true;
    }

    template <>
    bool FromPyObject(PyObject* obj, bool& res) {
        if (!PyBool_Check(obj))
            return false;
        if (obj == Py_True)
            res = true;
        else
            res = false;
        return true;
    }

    template <>
    bool FromPyObject(PyObject* obj, PyObject*& res) {
        Py_XINCREF(obj);
        res = obj;
        return true;
    }

    template <>
    bool FromPyObject(PyObject* obj, TPyObjectPtr& res) {
        res = TPyObjectPtr(obj);
        return true;
    }

    static inline bool _FromPyObject(PyObject* obj, TStringBuf& res) {
        char* str;
        Py_ssize_t len;
#if PY_MAJOR_VERSION >= 3
        if (PyUnicode_Check(obj)) {
            auto buf = PyUnicode_AsUTF8AndSize(obj, &len);
            res = TStringBuf(buf, len);
            return true;
        }
#endif
        if (-1 == PyBytes_AsStringAndSize(obj, &str, &len) || 0 > len)
            return false;
        res = TStringBuf(str, len);
        return true;
    }

    bool FromPyObject(PyObject* obj, TStringBuf& res) {
        return _FromPyObject(obj, res);
    }

    bool FromPyObject(PyObject* obj, TString& res) {
        TStringBuf str;
        if (!_FromPyObject(obj, str))
            return false;
        res = str;
        return true;
    }

    bool FromPyObject(PyObject* obj, TUtf16String& res) {
        if (!PyUnicode_Check(obj))
            return false;
        auto str = TPyObjectPtr(PyUnicode_AsUTF16String(obj), true);
        if (!str)
            return false;
        constexpr auto BOM_SIZE = 2;
        size_t len = (static_cast<size_t>(PyBytes_GET_SIZE(str.Get())) - BOM_SIZE) / 2;
        res.resize(len);
        memcpy(res.begin(), PyBytes_AS_STRING(str.Get()) + BOM_SIZE, len * 2);
        return (nullptr == PyErr_Occurred());
    }

    bool FromPyObject(PyObject* obj, TBuffer& res) {
        if (!PyList_Check(obj))
            return false;
        size_t cnt = PyList_Size(obj);
        res.Reserve(cnt);
        for (size_t i = 0; i < cnt; ++i) {
            PyObject* item = PyList_GET_ITEM(obj, i);
            char ch = 0;
            if (!FromPyObject(item, ch))
                return false;
            res.Append(ch);
        }
        return true;
    }
}
