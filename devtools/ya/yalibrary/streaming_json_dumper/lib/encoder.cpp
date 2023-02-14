#include "encoder.h"

#include <library/cpp/pybind/ptr.h>

#include <util/generic/buffer.h>
#include <util/generic/maybe.h>
#include <util/generic/scope.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/output.h>
#include <util/string/cast.h>
#include <util/stream/str.h>
#include <util/string/strspn.h>
#include <util/system/compiler.h>

#include <cmath>

using NPyBind::TPyObjectPtr;

namespace NSJson {

    namespace {
        constexpr bool BORROW = true;

        // Keep a Python object with Str together because Str refers to the object internal buffer
        struct TPythonUtf8String {
            TPyObjectPtr ObjStrRef;
            TStringBuf Str;
        };

        TMaybe<TPythonUtf8String> GetUtf8String(PyObject* obj) {
#if PY_MAJOR_VERSION == 2
            if (PyString_Check(obj)) {
                const char* data = PyString_AS_STRING(obj);
                Py_ssize_t len = PyString_GET_SIZE(obj);
                return {{TPyObjectPtr(obj), {data, static_cast<std::size_t>(len)}}};
            }
#endif
            if (PyUnicode_Check(obj)) {
#if (PY_VERSION_HEX >= 0x03030000)
                if (PyUnicode_IS_COMPACT_ASCII(obj)) {
                    Py_ssize_t len;
                    const char* data = PyUnicode_AsUTF8AndSize(obj, &len);
                    return {{TPyObjectPtr(obj), {data, static_cast<std::size_t>(len)}}};
                }
#endif
                TPyObjectPtr utf8Str{PyUnicode_AsUTF8String(obj), BORROW};
                if (!utf8Str.Get()) {
                    throw yexception() << "Cannot convert unicode to utf-8";
                }
#if PY_MAJOR_VERSION == 3
                const char* data = PyBytes_AS_STRING(utf8Str.Get());
                Py_ssize_t len = PyBytes_GET_SIZE(utf8Str.Get());
#else
                const char* data = PyString_AS_STRING(utf8Str.Get());
                Py_ssize_t len = PyString_GET_SIZE(utf8Str.Get());
#endif
                return {{std::move(utf8Str), {data, static_cast<std::size_t>(len)}}};
            }
            return {};
        }

        // str(obj)
        TString GetObjectStr(PyObject* obj) {
            TPyObjectPtr objStrRef{PyObject_Str(obj), BORROW};
            if (objStrRef.Get()) {
                TMaybe<TPythonUtf8String> str = GetUtf8String(objStrRef.Get());
                if (str.Defined()) {
                    return TString{str->Str};
                }
            }
            if (PyErr_Occurred()) {
                PyErr_Clear();
            }
            return "<Unknown object. Cannot represent as string>";
        }

        // Python stream.write() method wrapper
        class TPythonOutputStreamWrapper {
        public:
            TPythonOutputStreamWrapper(PyObject* stream)
                : WriteMethod_{PyObject_GetAttrString(stream, "write"), BORROW}
            {
                if (!WriteMethod_.Get()) {
                    throw TValueError() << "Stream doesn't have 'write' attribute";
                }
                if (!PyCallable_Check(WriteMethod_.Get())) {
                    throw TValueError() << "Stream 'write' attribute is not callable";
                }
            }

            void Write(const char* data, std::size_t len) {
                if (Y_LIKELY(len)) {
                    TPyObjectPtr buffer{PyBytes_FromStringAndSize(data, len), BORROW};
                    if (!buffer.Get()) {
                        throw yexception() << "Cannot create bytes object";
                    }
                    TPyObjectPtr result{PyObject_CallFunctionObjArgs(WriteMethod_.Get(), buffer.Get(), nullptr), BORROW};
                    if (!result.Get()) {
                        throw yexception() << "write() method failed";
                    }
                }
            }

        private:
            TPyObjectPtr WriteMethod_;
        };

        // This buffer is intended to reduce overhead of python write() method calling.
        // Default buffer size equals to the Linux pipe buffer size - optimal for UCompressor (devtools/ya/exts/compress.py).
        template<class TOutput, std::size_t BufSize = 65536>
        class TBufferedOutputStream {
        public:
            template <class... Args>
            TBufferedOutputStream(Args&&... args)
                : Output_{std::forward<Args>(args)...}
                , Buffer_{BufSize}
            {

            }

            void Write(const char* data, std::size_t len) {
                if (Buffer_.Avail() < len) {
                    Flush();
                }
                if (Y_LIKELY(Buffer_.Avail() >= len)) {
                    Buffer_.Append(data, len);
                } else {
                    Y_ASSERT(!Buffer_.Size());
                    Output_.Write(data, len);
                }
            }

            void Write(TStringBuf s) {
                Write(s.Data(), s.Size());
            }

            void Write(char ch){
                if (!Buffer_.Avail()) {
                    Flush();
                }
                Y_ASSERT(Buffer_.Avail());
                Buffer_.Append(ch);
            }

            void Flush() {
                if (Buffer_) {
                    Output_.Write(Buffer_.Data(), Buffer_.Size());
                    Buffer_.Clear();
                }
            }

        private:
            TOutput Output_;
            TBuffer Buffer_;
        };

        class TLoopDetector {
        public:
            TLoopDetector(std::size_t checkInterval = 1024)
                : CheckInterval_(checkInterval)
            {
            }

            void Enter(PyObject* obj) {
                Stack_.push_back(obj);
                if (Stack_.size() % CheckInterval_ == 0) {
                    if (auto b = Find(Stack_.begin(), Stack_.end() - 1, Stack_.back()); b != Stack_.end() - 1) {
                        // Make a descriptive error about the loop (see tests for examples).
                        TValueError error{};
                        error << "Circular reference found: ";
                        // Iterator b points to a random position inside the loop. Determine exact loop bounds.
                        // To get a loop size find the nearest element which equals to *b.
                        // Knowing the loop size it is easy to find a loop start.
                        auto e = Find(b + 1, Stack_.end(), *b);
                        Y_ASSERT(e != Stack_.end());
                        auto loopSize = e - b;
                        auto loopStart = Stack_.begin();
                        while (*loopStart != *(loopStart + loopSize)) {
                            AddIndexToError(error, loopStart);
                            ++loopStart;
                        }
                        error << "-->";
                        auto loopEnd = loopStart + loopSize;
                        for (auto pos = loopStart; pos != loopEnd; ++pos) {
                            AddIndexToError(error, pos);
                        }
                        error << "-->";
                        throw error;
                    }
                }
            }

            void Leave() {
                Stack_.pop_back();
            }

        private:
            void AddIndexToError(TValueError& error, TVector<PyObject*>::const_iterator item) {
                Y_ASSERT(item + 1 != Stack_.end());
                PyObject* obj = *item;
                PyObject* nextObj = *(item + 1);
                if (PyDict_Check(obj)) {
                    PyObject *key;
                    PyObject *value;
                    Py_ssize_t pos = 0;
                    while (PyDict_Next(obj, &pos, &key, &value)) {
                        if (value == nextObj) {
                            auto str = GetUtf8String(key);
                            Y_ASSERT(str.Defined());
                            error << "[\"" << str->Str << "\"]";
                            return;
                        }
                    }
                } else if (PyList_Check(obj)) {
                    Py_ssize_t listSize = PyList_Size(obj);
                    for (Py_ssize_t pos = 0; pos < listSize; ++pos) {
                        PyObject* value = PyList_GET_ITEM(obj, pos);
                        if (value == nextObj) {
                            error << "[" << pos << "]";
                            return;
                        }
                    }
                } else if (PyTuple_Check(obj)) {
                    Py_ssize_t tupleSize = PyTuple_Size(obj);
                    for (Py_ssize_t pos = 0; pos < tupleSize; ++pos) {
                        PyObject* value = PyTuple_GET_ITEM(obj, pos);
                        if (value == nextObj) {
                            error << "[" << pos << "]";
                            return;
                        }
                    }
                }
                Y_UNREACHABLE();
            }

        private:
            const std::size_t CheckInterval_;
            TVector<PyObject*> Stack_;
        };

        template<class TOut>
        class TJsonEncoder {
        public:
            TJsonEncoder(PyObject* stream)
                : StreamRef_{stream}
                , Out_{stream}
            {
                for (unsigned char c = 0x00; c < 0x20; ++c) {
                    EscapedChars_.Set(c);
                }
            }

            void Encode(PyObject* obj) {
                DoEncode(obj);
                Out_.Flush();
            }

        private:
            void DoEncode(PyObject* obj) {
                if (auto str = GetUtf8String(obj); str.Defined()) {
                    WriteEscapedString(str->Str);
                    return;
                }
                if (PyBool_Check(obj)) {
                    Out_.Write(obj == Py_True ? "true" : "false");
                    return;
                }
                if (PyLong_Check(obj)) {
                    long long val = PyLong_AsLongLong(obj);
                    if (!PyErr_Occurred()) {
                        Out_.Write(ToString(val));
                        return;
                    }
                    if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
                        PyErr_Clear();
                        unsigned long long uval = PyLong_AsUnsignedLongLong(obj);
                        if (!PyErr_Occurred()) {
                            Out_.Write(ToString(uval));
                            return;
                        }
                    }
                    PyErr_Clear();
                    throw TValueError() << "Cannot convert the following value to [unsigned] long long: " << GetObjectStr(obj);
                }
        #if PY_MAJOR_VERSION == 2
                if (PyInt_Check(obj)) {
                    long val = PyInt_AS_LONG(obj);
                    Out_.Write(ToString(val));
                    return;
                }
        #endif
                if (PyFloat_Check(obj)) {
                    double value = PyFloat_AsDouble(obj);
                    if (!PyErr_Occurred()) {
                        if (std::isnan(value) || std::isinf(value)) {
                            throw TValueError() << "Nan and Inf values are not permitted: " << GetObjectStr(obj);
                        }
                        TString s = FloatToString(value);
                        Out_.Write(s);
                        if (IsTrueFloat_.FindFirstOf(s.begin(), s.end()) == s.end()) {
                            Out_.Write(".0");
                        }
                        return;
                    }
                    PyErr_Clear();
                    throw TValueError() << "Cannot convert the following value to double: " << GetObjectStr(obj);
                }
                if (PyDict_Check(obj)) {
                    LoopDetector_.Enter(obj);
                    Y_SCOPE_EXIT(this) {LoopDetector_.Leave();};
                    TPyObjectPtr dictRef{obj};

                    Out_.Write('{');
                    bool needComma = false;

                    PyObject *key;
                    PyObject *value;
                    Py_ssize_t pos = 0;
                    while (PyDict_Next(obj, &pos, &key, &value)) {
                        if (Y_LIKELY(needComma)) {
                            Out_.Write(',');
                        } else {
                            needComma = true;
                        }
                        auto str = GetUtf8String(key);
                        if (str.Empty()) {
                            throw TValueError() << "Dict key is not a string: " << GetObjectStr(key);
                        }
                        WriteEscapedString(str->Str);
                        Out_.Write(':');
                        DoEncode(value);
                    }
                    Out_.Write('}');
                    return;
                }
                if (PyList_Check(obj)) {
                    LoopDetector_.Enter(obj);
                    Y_SCOPE_EXIT(this) {LoopDetector_.Leave();};
                    TPyObjectPtr listRef{obj};

                    Out_.Write('[');

                    Py_ssize_t listSize = PyList_Size(obj);
                    for (Py_ssize_t pos = 0; pos < listSize; ++pos) {
                        if (Y_LIKELY(pos)) {
                            Out_.Write(',');
                        }
                        PyObject* value = PyList_GET_ITEM(obj, pos);
                        DoEncode(value);
                    }
                    Out_.Write(']');
                    return;
                }
                if (PyTuple_Check(obj)) {
                    LoopDetector_.Enter(obj);
                    Y_SCOPE_EXIT(this) {LoopDetector_.Leave();};

                    Out_.Write('[');

                    Py_ssize_t tupleSize = PyTuple_Size(obj);
                    for (Py_ssize_t pos = 0; pos < tupleSize; ++pos) {
                        if (Y_LIKELY(pos)) {
                            Out_.Write(',');
                        }
                        PyObject* value = PyTuple_GET_ITEM(obj, pos);
                        DoEncode(value);
                    }
                    Out_.Write(']');
                    return;
                }
                if (obj == Py_None) {
                    Out_.Write("null");
                    return;
                }
                if (PyCallable_Check(obj)) {
                    TPyObjectPtr objRef{obj};
                    Out_.Flush();
                    TPyObjectPtr result{PyObject_CallFunctionObjArgs(obj, StreamRef_.Get(), nullptr), BORROW};
                    if (!result.Get()) {
                        throw yexception() << "External serialization method failed: " << GetObjectStr(obj);
                    }
                    return;
                }
                throw TValueError() << "Unsupported object: " << GetObjectStr(obj);
            }

            void WriteEscapedString(TStringBuf s) {
                static const char hexDigits[] = "0123456789abcdef";

                Out_.Write('"');

                const char* begin = s.Data();
                const char* end = begin + s.Size();
                const char* pos;
                while ((pos = EscapedChars_.FindFirstOf(begin, end)) != end) {
                    Out_.Write(begin, pos - begin);
                    unsigned char c = *pos;
                    switch (c) {
                        case '"':
                            Out_.Write("\\\"");
                            break;
                        case '\\':
                            Out_.Write("\\\\");
                            break;
                        case '\b':
                            Out_.Write("\\b");
                            break;
                        case '\f':
                            Out_.Write("\\f");
                            break;
                        case '\n':
                            Out_.Write("\\n");
                            break;
                        case '\r':
                            Out_.Write("\\r");
                            break;
                        case '\t':
                            Out_.Write("\\t");
                            break;
                        default:
                            Y_ASSERT(c < 0x20);
                            Out_.Write("\\u00");
                            Out_.Write(hexDigits[(c & 0xf0) >> 4]);
                            Out_.Write(hexDigits[c & 0x0f]);
                    }
                    begin = pos + 1;
                }
                Out_.Write(begin, end - begin);

                Out_.Write('"');
            }

        private:
            TPyObjectPtr StreamRef_;
            TOut Out_;
            TCompactStrSpn EscapedChars_{"\"\\\b\f\n\r\t"};
            TCompactStrSpn IsTrueFloat_{".Ee"};
            TLoopDetector LoopDetector_{};
        };
    }

    void Encode(PyObject* obj, PyObject* stream) {
        using TJsonOutputStream = TBufferedOutputStream<TPythonOutputStreamWrapper>;
        TPyObjectPtr objRef{obj};
        NSJson::TJsonEncoder<TJsonOutputStream> encoder(stream);
        encoder.Encode(obj);
    }
}
