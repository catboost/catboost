#include "json.h"

#include <library/cpp/json/json_value.h>

#include <util/string/cast.h>
#include <util/string/strspn.h>
#include <util/generic/algorithm.h>
#include <util/generic/ymath.h>
#include <util/generic/singleton.h>

namespace NJsonWriter {
    TBuf::TBuf(EHtmlEscapeMode mode, IOutputStream* stream)
        : Stream(stream)
        , NeedComma(false)
        , NeedNewline(false)
        , EscapeMode(mode)
        , IndentSpaces(0)
        , WriteNanAsString(false)
    {
        Y_ASSERT(mode == HEM_DONT_ESCAPE_HTML ||
                 mode == HEM_ESCAPE_HTML ||
                 mode == HEM_RELAXED ||
                 mode == HEM_UNSAFE);
        if (!Stream) {
            StringStream.Reset(new TStringStream);
            Stream = StringStream.Get();
        }

        Stack.reserve(64); // should be enough for most cases
        StackPush(JE_OUTER_SPACE);
    }

    static TStringBuf EntityToStr(EJsonEntity e) {
        switch (e) {
            case JE_OUTER_SPACE:
                return "JE_OUTER_SPACE";
            case JE_LIST:
                return "JE_LIST";
            case JE_OBJECT:
                return "JE_OBJECT";
            case JE_PAIR:
                return "JE_PAIR";
            default:
                return "JE_unknown";
        }
    }

    inline void TBuf::StackPush(EJsonEntity e) {
        Stack.push_back(e);
    }

    inline EJsonEntity TBuf::StackTop() const {
        return Stack.back();
    }

    inline void TBuf::StackPop() {
        Y_ASSERT(!Stack.empty());
        const EJsonEntity current = StackTop();
        Stack.pop_back();
        switch (current) {
            case JE_OUTER_SPACE:
                ythrow TError() << "JSON writer: stack empty";
            case JE_LIST:
                PrintIndentation(true);
                RawWriteChar(']');
                break;
            case JE_OBJECT:
                PrintIndentation(true);
                RawWriteChar('}');
                break;
            case JE_PAIR:
                break;
        }
        NeedComma = true;
        NeedNewline = true;
    }

    inline void TBuf::CheckAndPop(EJsonEntity e) {
        if (Y_UNLIKELY(StackTop() != e)) {
            ythrow TError() << "JSON writer: unexpected value "
                            << EntityToStr(StackTop()) << " on the stack";
        }
        StackPop();
    }

    void TBuf::PrintIndentation(bool closing) {
        if (!IndentSpaces)
            return;
        const int indentation = IndentSpaces * (Stack.size() - 1);
        if (!indentation && !closing)
            return;

        PrintWhitespaces(Max(0, indentation), true);
    }

    void TBuf::PrintWhitespaces(size_t count, bool prependWithNewLine) {
        static constexpr TStringBuf whitespacesTemplate = "\n                                ";
        static_assert(whitespacesTemplate[0] == '\n');
        static_assert(whitespacesTemplate[1] == ' ');

        count += (prependWithNewLine);
        do {
            const TStringBuf buffer = whitespacesTemplate.SubString(prependWithNewLine ? 0 : 1, count);
            count -= buffer.size();
            UnsafeWriteRawBytes(buffer);
            prependWithNewLine = false;  // skip '\n' in subsequent writes
        } while (count > 0);
    }

    inline void TBuf::WriteComma() {
        if (NeedComma) {
            RawWriteChar(',');
        }
        NeedComma = true;

        if (NeedNewline) {
            PrintIndentation(false);
        }
        NeedNewline = true;
    }

    inline void TBuf::BeginValue() {
        if (Y_UNLIKELY(KeyExpected())) {
            ythrow TError() << "JSON writer: value written, "
                               "but expected a key:value pair";
        }
        WriteComma();
    }

    inline void TBuf::BeginKey() {
        if (Y_UNLIKELY(!KeyExpected())) {
            ythrow TError() << "JSON writer: key written outside of an object";
        }
        WriteComma();
        StackPush(JE_PAIR);
        NeedComma = false;
        NeedNewline = false;
    }

    inline void TBuf::EndValue() {
        if (StackTop() == JE_PAIR) {
            StackPop();
        }
    }

    TValueContext TBuf::BeginList() {
        NeedNewline = true;
        BeginValue();
        RawWriteChar('[');
        StackPush(JE_LIST);
        NeedComma = false;
        return TValueContext(*this);
    }

    TPairContext TBuf::BeginObject() {
        NeedNewline = true;
        BeginValue();
        RawWriteChar('{');
        StackPush(JE_OBJECT);
        NeedComma = false;
        return TPairContext(*this);
    }

    TAfterColonContext TBuf::UnsafeWriteKey(const TStringBuf& s) {
        BeginKey();
        RawWriteChar('"');
        UnsafeWriteRawBytes(s);
        UnsafeWriteRawBytes("\":", 2);
        return TAfterColonContext(*this);
    }

    TAfterColonContext TBuf::WriteKey(const TStringBuf& s) {
        // use the default escaping mode for this object
        return WriteKey(s, EscapeMode);
    }

    TAfterColonContext TBuf::WriteKey(const TStringBuf& s, EHtmlEscapeMode hem) {
        BeginKey();
        WriteBareString(s, hem);
        RawWriteChar(':');
        return TAfterColonContext(*this);
    }

    TAfterColonContext TBuf::CompatWriteKeyWithoutQuotes(const TStringBuf& s) {
        BeginKey();
        Y_ASSERT(AllOf(s, [](char x) { return 'a' <= x && x <= 'z'; }));
        UnsafeWriteRawBytes(s);
        RawWriteChar(':');
        return TAfterColonContext(*this);
    }

    TBuf& TBuf::EndList() {
        CheckAndPop(JE_LIST);
        EndValue();
        return *this;
    }

    TBuf& TBuf::EndObject() {
        CheckAndPop(JE_OBJECT);
        EndValue();
        return *this;
    }

    TValueContext TBuf::WriteString(const TStringBuf& s) {
        // use the default escaping mode for this object
        return WriteString(s, EscapeMode);
    }

    TValueContext TBuf::WriteString(const TStringBuf& s, EHtmlEscapeMode hem) {
        BeginValue();
        WriteBareString(s, hem);
        EndValue();
        return TValueContext(*this);
    }

    TValueContext TBuf::WriteNull() {
        UnsafeWriteValue(TStringBuf("null"));
        return TValueContext(*this);
    }

    TValueContext TBuf::WriteBool(bool b) {
        constexpr TStringBuf trueVal = "true";
        constexpr TStringBuf falseVal = "false";
        UnsafeWriteValue(b ? trueVal : falseVal);
        return TValueContext(*this);
    }

    TValueContext TBuf::WriteInt(int i) {
        char buf[22]; // enough to hold any 64-bit number
        size_t len = ToString(i, buf, sizeof(buf));
        UnsafeWriteValue(buf, len);
        return TValueContext(*this);
    }

    TValueContext TBuf::WriteLongLong(long long i) {
        static_assert(sizeof(long long) <= 8, "expect sizeof(long long) <= 8");
        char buf[22]; // enough to hold any 64-bit number
        size_t len = ToString(i, buf, sizeof(buf));
        UnsafeWriteValue(buf, len);
        return TValueContext(*this);
    }

    TValueContext TBuf::WriteULongLong(unsigned long long i) {
        char buf[22]; // enough to hold any 64-bit number
        size_t len = ToString(i, buf, sizeof(buf));
        UnsafeWriteValue(buf, len);
        return TValueContext(*this);
    }

    template <class TFloat>
    TValueContext TBuf::WriteFloatImpl(TFloat f, EFloatToStringMode mode, int ndigits) {
        char buf[512]; // enough to hold most floats, the same buffer is used in FloatToString implementation
        if (Y_UNLIKELY(!IsValidFloat(f))) {
            if (WriteNanAsString) {
                const size_t size = FloatToString(f, buf, Y_ARRAY_SIZE(buf));
                WriteString(TStringBuf(buf, size));
                return TValueContext(*this);
            } else {
                ythrow TError() << "JSON writer: invalid float value: " << FloatToString(f);
            }
        }
        size_t len = FloatToString(f, buf, Y_ARRAY_SIZE(buf), mode, ndigits);
        UnsafeWriteValue(buf, len);
        return TValueContext(*this);
    }

    TValueContext TBuf::WriteFloat(float f, EFloatToStringMode mode, int ndigits) {
        return WriteFloatImpl(f, mode, ndigits);
    }

    TValueContext TBuf::WriteDouble(double f, EFloatToStringMode mode, int ndigits) {
        return WriteFloatImpl(f, mode, ndigits);
    }

    namespace {
        struct TFinder: public TCompactStrSpn {
            inline TFinder()
                : TCompactStrSpn("\xe2\\\"\b\n\f\r\t<>&\'/")
            {
                for (ui8 ch = 0; ch < 0x20; ++ch) {
                    Set(ch);
                }
            }
        };
    }

    inline void TBuf::WriteBareString(const TStringBuf s, EHtmlEscapeMode hem) {
        RawWriteChar('"');
        const auto& specialChars = *Singleton<TFinder>();
        const char* b = s.begin();
        const char* e = s.end();
        const char* i = b;
        while ((i = specialChars.FindFirstOf(i, e)) != e) {
            // U+2028 (line separator) and U+2029 (paragraph separator) are valid string
            // contents in JSON, but are treated as line breaks in JavaScript, breaking JSONP.
            // In UTF-8, U+2028 is "\xe2\x80\xa8" and U+2029 is "\xe2\x80\xa9".
            if (Y_UNLIKELY(e - i >= 3 && i[0] == '\xe2' && i[1] == '\x80' && (i[2] | 1) == '\xa9')) {
                UnsafeWriteRawBytes(b, i - b);
                UnsafeWriteRawBytes(i[2] == '\xa9' ? "\\u2029" : "\\u2028", 6);
                b = i = i + 3;
            } else if (EscapedWriteChar(b, i, hem)) {
                b = ++i;
            } else {
                ++i;
            }
        }
        UnsafeWriteRawBytes(b, e - b);
        RawWriteChar('"');
    }

    inline void TBuf::RawWriteChar(char c) {
        Stream->Write(c);
    }

    void TBuf::WriteHexEscape(unsigned char c) {
        Y_ASSERT(c < 0x80);
        UnsafeWriteRawBytes("\\u00", 4);
        static const char hexDigits[] = "0123456789ABCDEF";
        RawWriteChar(hexDigits[(c & 0xf0) >> 4]);
        RawWriteChar(hexDigits[(c & 0x0f)]);
    }

#define MATCH(sym, string)                        \
    case sym:                                     \
        UnsafeWriteRawBytes(beg, cur - beg);      \
        UnsafeWriteRawBytes(TStringBuf(string));  \
        return true

    inline bool TBuf::EscapedWriteChar(const char* beg, const char* cur, EHtmlEscapeMode hem) {
        unsigned char c = *cur;
        if (hem == HEM_ESCAPE_HTML) {
            switch (c) {
                MATCH('"', "&quot;");
                MATCH('\'', "&#39;");
                MATCH('<', "&lt;");
                MATCH('>', "&gt;");
                MATCH('&', "&amp;");
            }
            //for other characters, we fall through to the non-HTML-escaped part
        }

        if (hem == HEM_RELAXED && c == '/')
            return false;

        if (hem != HEM_UNSAFE) {
            switch (c) {
                case '/':
                    UnsafeWriteRawBytes(beg, cur - beg);
                    UnsafeWriteRawBytes("\\/", 2);
                    return true;
                case '<':
                case '>':
                case '\'':
                    UnsafeWriteRawBytes(beg, cur - beg);
                    WriteHexEscape(c);
                    return true;
            }
            // for other characters, fall through to the non-escaped part
        }

        switch (c) {
            MATCH('"', "\\\"");
            MATCH('\\', "\\\\");
            MATCH('\b', "\\b");
            MATCH('\f', "\\f");
            MATCH('\n', "\\n");
            MATCH('\r', "\\r");
            MATCH('\t', "\\t");
        }
        if (c < 0x20) {
            UnsafeWriteRawBytes(beg, cur - beg);
            WriteHexEscape(c);
            return true;
        }

        return false;
    }

#undef MATCH

    static bool LessStrPtr(const TString* a, const TString* b) {
        return *a < *b;
    }

    TValueContext TBuf::WriteJsonValue(const NJson::TJsonValue* v, bool sortKeys, EFloatToStringMode mode, int ndigits) {
        using namespace NJson;
        switch (v->GetType()) {
            default:
            case JSON_NULL:
                WriteNull();
                break;
            case JSON_BOOLEAN:
                WriteBool(v->GetBoolean());
                break;
            case JSON_DOUBLE:
                WriteDouble(v->GetDouble(), mode, ndigits);
                break;
            case JSON_INTEGER:
                WriteLongLong(v->GetInteger());
                break;
            case JSON_UINTEGER:
                WriteULongLong(v->GetUInteger());
                break;
            case JSON_STRING:
                WriteString(v->GetString());
                break;
            case JSON_ARRAY: {
                BeginList();
                const TJsonValue::TArray& arr = v->GetArray();
                for (const auto& it : arr)
                    WriteJsonValue(&it, sortKeys, mode, ndigits);
                EndList();
                break;
            }
            case JSON_MAP: {
                BeginObject();
                const TJsonValue::TMapType& map = v->GetMap();
                if (sortKeys) {
                    const size_t oldsz = Keys.size();
                    Keys.reserve(map.size() + oldsz);
                    for (const auto& it : map) {
                        Keys.push_back(&(it.first));
                    }
                    Sort(Keys.begin() + oldsz, Keys.end(), LessStrPtr);
                    for (size_t i = oldsz, sz = Keys.size(); i < sz; ++i) {
                        TJsonValue::TMapType::const_iterator kv = map.find(*Keys[i]);
                        WriteKey(kv->first);
                        WriteJsonValue(&kv->second, sortKeys, mode, ndigits);
                    }
                    Keys.resize(oldsz);
                } else {
                    for (const auto& it : map) {
                        WriteKey(it.first);
                        WriteJsonValue(&it.second, sortKeys, mode, ndigits);
                    }
                }
                EndObject();
                break;
            }
        }
        return TValueContext(*this);
    }

    TPairContext TBuf::UnsafeWritePair(const TStringBuf& s) {
        if (Y_UNLIKELY(StackTop() != JE_OBJECT)) {
            ythrow TError() << "JSON writer: key:value pair written outside of an object";
        }
        WriteComma();
        UnsafeWriteRawBytes(s);
        return TPairContext(*this);
    }

    void TBuf::UnsafeWriteValue(const TStringBuf& s) {
        BeginValue();
        UnsafeWriteRawBytes(s);
        EndValue();
    }

    void TBuf::UnsafeWriteValue(const char* s, size_t len) {
        BeginValue();
        UnsafeWriteRawBytes(s, len);
        EndValue();
    }

    void TBuf::UnsafeWriteRawBytes(const char* src, size_t len) {
        Stream->Write(src, len);
    }

    void TBuf::UnsafeWriteRawBytes(const TStringBuf& s) {
        UnsafeWriteRawBytes(s.data(), s.size());
    }

    const TString& TBuf::Str() const {
        if (!StringStream) {
            ythrow TError() << "JSON writer: Str() called "
                               "but writing to an external stream";
        }
        if (!(Stack.size() == 1 && StackTop() == JE_OUTER_SPACE)) {
            ythrow TError() << "JSON writer: incomplete object converted to string";
        }
        return StringStream->Str();
    }

    void TBuf::FlushTo(IOutputStream* stream) {
        if (!StringStream) {
            ythrow TError() << "JSON writer: FlushTo() called "
                               "but writing to an external stream";
        }
        stream->Write(StringStream->Str());
        StringStream->Clear();
    }

    TString WrapJsonToCallback(const TBuf& buf, TStringBuf callback) {
        if (!callback) {
            return buf.Str();
        } else {
            return TString::Join(callback, "(", buf.Str(), ")");
        }
    }

    TBufState TBuf::State() const {
        return TBufState{NeedComma, NeedNewline, Stack};
    }

    void TBuf::Reset(const TBufState& from) {
        NeedComma = from.NeedComma;
        NeedNewline = from.NeedNewline;
        Stack = from.Stack;
    }

    void TBuf::Reset(TBufState&& from) {
        NeedComma = from.NeedComma;
        NeedNewline = from.NeedNewline;
        Stack.swap(from.Stack);
    }

}
