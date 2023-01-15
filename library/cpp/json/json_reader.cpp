#include "json_reader.h"

#include "rapidjson_helpers.h"

#include <contrib/libs/rapidjson/include/rapidjson/document.h>
#include <contrib/libs/rapidjson/include/rapidjson/error/en.h>
#include <contrib/libs/rapidjson/include/rapidjson/error/error.h>
#include <contrib/libs/rapidjson/include/rapidjson/reader.h>
#include <contrib/libs/rapidjson/include/rapidjson/stringbuffer.h>
#include <contrib/libs/rapidjson/include/rapidjson/writer.h>

#include <util/generic/stack.h>
#include <util/string/cast.h>
#include <util/system/yassert.h>
#include <util/string/builder.h>

namespace NJson {
    namespace {
        TString PrintError(const rapidjson::ParseResult& result) {
            return TStringBuilder() << AsStringBuf("Offset: ") << result.Offset()
                                    << AsStringBuf(", Code: ") << (int)result.Code()
                                    << AsStringBuf(", Error: ") << GetParseError_En(result.Code());
        }
    }

    static const size_t DEFAULT_BUFFER_LEN = 65536;

    bool TParserCallbacks::OpenComplexValue(EJsonValueType type) {
        TJsonValue* pvalue;
        switch (CurrentState) {
            case START:
                Value.SetType(type);
                ValuesStack.push_back(&Value);
                break;
            case IN_ARRAY:
                pvalue = &ValuesStack.back()->AppendValue(type);
                ValuesStack.push_back(pvalue);
                break;
            case AFTER_MAP_KEY:
                pvalue = &ValuesStack.back()->InsertValue(Key, type);
                ValuesStack.push_back(pvalue);
                CurrentState = IN_MAP;
                break;
            default:
                return false;
        }
        return true;
    }

    bool TParserCallbacks::CloseComplexValue() {
        if (ValuesStack.empty()) {
            return false;
        }

        ValuesStack.pop_back();
        if (!ValuesStack.empty()) {
            switch (ValuesStack.back()->GetType()) {
                case JSON_ARRAY:
                    CurrentState = IN_ARRAY;
                    break;
                case JSON_MAP:
                    CurrentState = IN_MAP;
                    break;
                default:
                    return false;
            }
        } else {
            CurrentState = FINISH;
        }
        return true;
    }

    TParserCallbacks::TParserCallbacks(TJsonValue& value, bool throwOnError, bool notClosedBracketIsError)
        : TJsonCallbacks(throwOnError)
        , Value(value)
        , NotClosedBracketIsError(notClosedBracketIsError)
        , CurrentState(START)
    {
    }

    bool TParserCallbacks::OnNull() {
        return SetValue(JSON_NULL);
    }

    bool TParserCallbacks::OnBoolean(bool val) {
        return SetValue(val);
    }

    bool TParserCallbacks::OnInteger(long long val) {
        return SetValue(val);
    }

    bool TParserCallbacks::OnUInteger(unsigned long long val) {
        return SetValue(val);
    }

    bool TParserCallbacks::OnString(const TStringBuf& val) {
        return SetValue(val);
    }

    bool TParserCallbacks::OnDouble(double val) {
        return SetValue(val);
    }

    bool TParserCallbacks::OnOpenArray() {
        bool res = OpenComplexValue(JSON_ARRAY);
        if (res)
            CurrentState = IN_ARRAY;
        return res;
    }

    bool TParserCallbacks::OnCloseArray() {
        return CloseComplexValue();
    }

    bool TParserCallbacks::OnOpenMap() {
        bool res = OpenComplexValue(JSON_MAP);
        if (res)
            CurrentState = IN_MAP;
        return res;
    }

    bool TParserCallbacks::OnCloseMap() {
        return CloseComplexValue();
    }

    bool TParserCallbacks::OnMapKey(const TStringBuf& val) {
        switch (CurrentState) {
            case IN_MAP:
                Key = val;
                CurrentState = AFTER_MAP_KEY;
                break;
            default:
                return false;
        }
        return true;
    }

    bool TParserCallbacks::OnEnd() {
        if (NotClosedBracketIsError){
            return ValuesStack.empty();
        }
        return true;
    }

    TJsonReaderConfig::TJsonReaderConfig()
        : BufferSize(DEFAULT_BUFFER_LEN)
    {
    }

    void TJsonReaderConfig::SetBufferSize(size_t bufferSize) {
        BufferSize = Max((size_t)1, Min(bufferSize, DEFAULT_BUFFER_LEN));
    }

    size_t TJsonReaderConfig::GetBufferSize() const {
        return BufferSize;
    }

    namespace {
        struct TJsonValueBuilder {
#ifdef NDEBUG
            using TItem = TJsonValue*;

            inline TJsonValue& Access(TItem& item) const {
                return *item;
            }
#else
            struct TItem {
                TJsonValue* V;
                size_t DuplicateKeyCount;

                TItem(TJsonValue* v)
                    : V(v)
                    , DuplicateKeyCount(0)
                {
                }
            };

            inline TJsonValue& Access(TItem& item) const {
                return *item.V;
            }
#endif

            NJson::TJsonValue& V;

            TStack<TItem> S;

            TJsonValueBuilder(NJson::TJsonValue& v)
                : V(v)
            {
                S.emplace(&V);
            }

            template <class T>
            void Set(const T& t) {
                if (Access(S.top()).IsArray()) {
                    Access(S.top()).AppendValue(t);
                } else {
                    Access(S.top()) = t;
                    S.pop();
                }
            }

            bool Null() {
                Set(NJson::JSON_NULL);
                return true;
            }

            bool Bool(bool b) {
                Set(b);
                return true;
            }

            bool Int(int i) {
                Set(i);
                return true;
            }

            template <class U>
            bool ProcessUint(U u) {
                if (Y_LIKELY(u <= static_cast<ui64>(Max<i64>()))) {
                    Set(i64(u));
                } else {
                    Set(u);
                }
                return true;
            }

            bool Uint(unsigned u) {
                return ProcessUint(u);
            }

            bool Int64(i64 i) {
                Set(i);
                return true;
            }

            bool Uint64(ui64 u) {
                return ProcessUint(u);
            }

            bool Double(double d) {
                Set(d);
                return true;
            }

            bool RawNumber(const char* str, rapidjson::SizeType length, bool copy) {
                Y_ASSERT(false && "this method should never be called");
                Y_UNUSED(str);
                Y_UNUSED(length);
                Y_UNUSED(copy);
                return true;
            }

            bool String(const char* str, rapidjson::SizeType length, bool copy) {
                Y_ASSERT(copy);
                Set(TStringBuf(str, length));
                return true;
            }

            bool StartObject() {
                if (Access(S.top()).IsArray()) {
                    S.emplace(&Access(S.top()).AppendValue(NJson::JSON_MAP));
                } else {
                    Access(S.top()).SetType(NJson::JSON_MAP);
                }
                return true;
            }

            bool Key(const char* str, rapidjson::SizeType length, bool copy) {
                Y_ASSERT(copy);
                auto& value = Access(S.top())[TStringBuf(str, length)];
                if (Y_UNLIKELY(value.GetType() != JSON_UNDEFINED)) {
#ifndef NDEBUG
                    ++S.top().DuplicateKeyCount;
#endif
                    value.SetType(JSON_UNDEFINED);
                }
                S.emplace(&value);
                return true;
            }

            inline int GetDuplicateKeyCount() const {
#ifdef NDEBUG
                return 0;
#else
                return S.top().DuplicateKeyCount;
#endif
            }

            bool EndObject(rapidjson::SizeType memberCount) {
                Y_ASSERT(memberCount == Access(S.top()).GetMap().size() + GetDuplicateKeyCount());
                S.pop();
                return true;
            }

            bool StartArray() {
                if (Access(S.top()).IsArray()) {
                    S.emplace(&Access(S.top()).AppendValue(NJson::JSON_ARRAY));
                } else {
                    Access(S.top()).SetType(NJson::JSON_ARRAY);
                }
                return true;
            }

            bool EndArray(rapidjson::SizeType elementCount) {
                Y_ASSERT(elementCount == Access(S.top()).GetArray().size());
                S.pop();
                return true;
            }
        };

        template <class TRapidJsonCompliantInputStream, class THandler>
        auto Read(const TJsonReaderConfig& config,
                  rapidjson::Reader& reader,
                  TRapidJsonCompliantInputStream& is,
                  THandler& handler) {

            ui8 flags = ReaderConfigToRapidJsonFlags::NOCOMMENTS_VALID_NOESCAPE;

            if (config.AllowComments) {
                flags |= ReaderConfigFlags::COMMENTS;
            }

            if (config.DontValidateUtf8) {
                flags &= ~(ReaderConfigFlags::VALIDATE);
            }

            if (config.AllowEscapedApostrophe) {
                flags |= ReaderConfigFlags::ESCAPE;
            }

            switch (flags) {
                case ReaderConfigToRapidJsonFlags::COMMENTS_NOVALID_NOESCAPE:
                    return reader.Parse<rapidjson::kParseCommentsFlag>(is, handler);
                case ReaderConfigToRapidJsonFlags::COMMENTS_VALID_NOESCAPE:
                    return reader.Parse<rapidjson::kParseCommentsFlag | rapidjson::kParseValidateEncodingFlag>(is, handler);
                case ReaderConfigToRapidJsonFlags::COMMENTS_VALID_ESCAPE:
                    return reader.Parse<rapidjson::kParseCommentsFlag | rapidjson::kParseValidateEncodingFlag | rapidjson::kParseEscapedApostropheFlag>(is, handler);
                case ReaderConfigToRapidJsonFlags::COMMENTS_NOVALID_ESCAPE:
                    return reader.Parse<rapidjson::kParseCommentsFlag | rapidjson::kParseEscapedApostropheFlag>(is, handler);
                case ReaderConfigToRapidJsonFlags::NOCOMMENTS_VALID_NOESCAPE:
                    return reader.Parse<rapidjson::kParseValidateEncodingFlag>(is, handler);
                case ReaderConfigToRapidJsonFlags::NOCOMMENTS_VALID_ESCAPE:
                    return reader.Parse<rapidjson::kParseValidateEncodingFlag | rapidjson::kParseEscapedApostropheFlag>(is, handler);
                case  ReaderConfigToRapidJsonFlags::NOCOMMENTS_NOVALID_ESCAPE:
                    return reader.Parse<rapidjson::kParseEscapedApostropheFlag>(is, handler);
                default:
                    return reader.Parse<rapidjson::kParseNoFlags>(is, handler);
            }
        }

        template <class TRapidJsonCompliantInputStream, class THandler>
        bool ReadJson(TRapidJsonCompliantInputStream& is, const TJsonReaderConfig* config, THandler& handler, bool throwOnError) {
            rapidjson::Reader reader;

            auto result = Read(*config, reader, is, handler);

            if (result.IsError()) {
                if (throwOnError) {
                    ythrow TJsonException() << PrintError(result);
                } else {
                    return false;
                }
            }

            return true;
        }

        template <class TRapidJsonCompliantInputStream>
        bool ReadJsonTree(TRapidJsonCompliantInputStream& is, const TJsonReaderConfig* config, TJsonValue* out, bool throwOnError) {
            out->SetType(NJson::JSON_NULL);

            TJsonValueBuilder handler(*out);

            return ReadJson(is, config, handler, throwOnError);
        }

        template <class TData>
        bool ReadJsonTreeImpl(TData* in, const TJsonReaderConfig* config, TJsonValue* out, bool throwOnError) {
            std::conditional_t<std::is_same<TData, TStringBuf>::value, TStringBufStreamWrapper, TInputStreamWrapper> is(*in);
            return ReadJsonTree(is, config, out, throwOnError);
        }

        template <class TData>
        bool ReadJsonTreeImpl(TData* in, bool allowComments, TJsonValue* out, bool throwOnError) {
            TJsonReaderConfig config;
            config.AllowComments = allowComments;
            return ReadJsonTreeImpl(in, &config, out, throwOnError);
        }

        template <class TData>
        bool ReadJsonTreeImpl(TData* in, TJsonValue* out, bool throwOnError) {
            return ReadJsonTreeImpl(in, false, out, throwOnError);
        }
    } //namespace

    bool ReadJsonTree(TStringBuf in, TJsonValue* out, bool throwOnError) {
        return ReadJsonTreeImpl(&in, out, throwOnError);
    }

    bool ReadJsonTree(TStringBuf in, bool allowComments, TJsonValue* out, bool throwOnError) {
        return ReadJsonTreeImpl(&in, allowComments, out, throwOnError);
    }

    bool ReadJsonTree(TStringBuf in, const TJsonReaderConfig* config, TJsonValue* out, bool throwOnError) {
        return ReadJsonTreeImpl(&in, config, out, throwOnError);
    }

    bool ReadJsonTree(IInputStream* in, TJsonValue* out, bool throwOnError) {
        return ReadJsonTreeImpl(in, out, throwOnError);
    }

    bool ReadJsonTree(IInputStream* in, bool allowComments, TJsonValue* out, bool throwOnError) {
        return ReadJsonTreeImpl(in, allowComments, out, throwOnError);
    }

    bool ReadJsonTree(IInputStream* in, const TJsonReaderConfig* config, TJsonValue* out, bool throwOnError) {
        return ReadJsonTreeImpl(in, config, out, throwOnError);
    }

    bool ReadJsonFastTree(TStringBuf in, TJsonValue* out, bool throwOnError, bool notClosedBracketIsError) {
        TParserCallbacks cb(*out, throwOnError, notClosedBracketIsError);

        return ReadJsonFast(in, &cb);
    }

    TJsonValue ReadJsonFastTree(TStringBuf in, bool notClosedBracketIsError) {
        TJsonValue value;
        // There is no way to report an error apart from throwing an exception when we return result by value.
        ReadJsonFastTree(in, &value, /* throwOnError = */ true, notClosedBracketIsError);
        return value;
    }

    namespace {
        struct TJsonCallbacksWrapper {
            TJsonCallbacks& Impl;

            TJsonCallbacksWrapper(TJsonCallbacks& impl)
                : Impl(impl)
            {
            }

            bool Null() {
                return Impl.OnNull();
            }

            bool Bool(bool b) {
                return Impl.OnBoolean(b);
            }

            template <class U>
            bool ProcessUint(U u) {
                if (Y_LIKELY(u <= ui64(Max<i64>()))) {
                    return Impl.OnInteger(i64(u));
                } else {
                    return Impl.OnUInteger(u);
                }
            }

            bool Int(int i) {
                return Impl.OnInteger(i);
            }

            bool Uint(unsigned u) {
                return ProcessUint(u);
            }

            bool Int64(i64 i) {
                return Impl.OnInteger(i);
            }

            bool Uint64(ui64 u) {
                return ProcessUint(u);
            }

            bool Double(double d) {
                return Impl.OnDouble(d);
            }

            bool RawNumber(const char* str, rapidjson::SizeType length, bool copy) {
                Y_ASSERT(false && "this method should never be called");
                Y_UNUSED(str);
                Y_UNUSED(length);
                Y_UNUSED(copy);
                return true;
            }

            bool String(const char* str, rapidjson::SizeType length, bool copy) {
                Y_ASSERT(copy);
                return Impl.OnString(TStringBuf(str, length));
            }

            bool StartObject() {
                return Impl.OnOpenMap();
            }

            bool Key(const char* str, rapidjson::SizeType length, bool copy) {
                Y_ASSERT(copy);
                return Impl.OnMapKey(TStringBuf(str, length));
            }

            bool EndObject(rapidjson::SizeType memberCount) {
                Y_UNUSED(memberCount);
                return Impl.OnCloseMap();
            }

            bool StartArray() {
                return Impl.OnOpenArray();
            }

            bool EndArray(rapidjson::SizeType elementCount) {
                Y_UNUSED(elementCount);
                return Impl.OnCloseArray();
            }
        };
    }

    bool ReadJson(IInputStream* in, TJsonCallbacks* cbs) {
        return ReadJson(in, false, cbs);
    }

    bool ReadJson(IInputStream* in, bool allowComments, TJsonCallbacks* cbs) {
        TJsonReaderConfig config;
        config.AllowComments = allowComments;
        return ReadJson(in, &config, cbs);
    }

    bool ReadJson(IInputStream* in, bool allowComments, bool allowEscapedApostrophe, TJsonCallbacks* cbs) {
        TJsonReaderConfig config;
        config.AllowComments = allowComments;
        config.AllowEscapedApostrophe = allowEscapedApostrophe;
        return ReadJson(in, &config, cbs);
    }

    bool ReadJson(IInputStream* in, const TJsonReaderConfig* config, TJsonCallbacks* cbs) {
        TJsonCallbacksWrapper wrapper(*cbs);
        TInputStreamWrapper is(*in);

        rapidjson::Reader reader;
        auto result = Read(*config, reader, is, wrapper);

        if (result.IsError()) {
            cbs->OnError(result.Offset(), PrintError(result));

            return false;
        }

        return cbs->OnEnd();
    }

    TJsonValue ReadJsonTree(IInputStream* in, bool throwOnError) {
        TJsonValue out;
        ReadJsonTree(in, &out, throwOnError);
        return out;
    }

    TJsonValue ReadJsonTree(IInputStream* in, bool allowComments, bool throwOnError) {
        TJsonValue out;
        ReadJsonTree(in, allowComments, &out, throwOnError);
        return out;
    }

    TJsonValue ReadJsonTree(IInputStream* in, const TJsonReaderConfig* config, bool throwOnError) {
        TJsonValue out;
        ReadJsonTree(in, config, &out, throwOnError);
        return out;
    }

}
