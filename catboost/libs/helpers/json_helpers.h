#pragma once

#include "exception.h"
#include <library/cpp/json/json_value.h>
#include <library/cpp/json/json_writer.h>

#include <util/stream/str.h>
#include <util/system/compiler.h>


template <typename T>
static NJson::TJsonValue VectorToJson(const TVector<T>& values) {
    NJson::TJsonValue jsonValue;
    jsonValue.SetType(NJson::EJsonValueType::JSON_ARRAY);
    for (const auto& value: values) {
        jsonValue.AppendValue(NJson::TJsonValue(value));
    }
    CB_ENSURE(jsonValue.GetArray().size() == values.size());
    return jsonValue;
}

static void WriteJsonWithCatBoostPrecision(
    const NJson::TJsonValue& value,
    bool formatOutput,
    IOutputStream* out
) {
    NJson::TJsonWriterConfig config;
    config.FormatOutput = formatOutput;
    config.FloatNDigits = 9;
    config.DoubleNDigits = 17;
    config.SortKeys = true;
    WriteJson(out, &value, config);
}

inline TString WriteJsonWithCatBoostPrecision(const NJson::TJsonValue& value, bool formatOutput) {
    TStringStream ss;
    WriteJsonWithCatBoostPrecision(value, formatOutput, &ss);
    return ss.Str();
}

template <typename T>
void FromJson(const NJson::TJsonValue& value, T* result) {
    switch (value.GetType()) {
        case NJson::EJsonValueType::JSON_INTEGER:
            *result = T(value.GetInteger());
            break;
        case NJson::EJsonValueType::JSON_DOUBLE:
            *result = T(value.GetDouble());
            break;
        case NJson::EJsonValueType::JSON_UINTEGER:
            *result = T(value.GetUInteger());
            break;
        case NJson::EJsonValueType::JSON_STRING:
            *result = FromString<T>(value.GetString());
            break;
        default:
            CB_ENSURE("Incorrect format");
    }
}

void FromJson(const NJson::TJsonValue& value, TString* result);

template <typename T>
static T FromJson(const NJson::TJsonValue& value) {
    T result;
    FromJson(value, &result);
    return result;
}

template <typename T>
static TVector<T> JsonToVector(const NJson::TJsonValue& jsonValue) {
    TVector<T> result;
    for (const auto& value: jsonValue.GetArray()) {
        result.push_back(FromJson<T>(value));
    }
    return result;
}

NJson::TJsonValue ReadTJsonValue(TStringBuf paramsJson);

/*
 * Use this function instead of simple ToString(jsonValue) because it saves floating point values with proper precision
 */
TString WriteTJsonValue(const NJson::TJsonValue& jsonValue);


template <typename T>
static void InsertEnumType(const TString& typeName, const T& value, NJson::TJsonValue* jsonValuePtr) {
    jsonValuePtr->InsertValue(typeName, ToString<T>(value));
}

template <typename T>
static void ReadEnumType(const TString& typeName, const NJson::TJsonValue& jsonValue, T* valuePtr) {
    *valuePtr = FromString<T>(jsonValue[typeName].GetString());
}


template <class T, bool IsEnum = std::is_enum<T>::value>
class TJsonFieldHelper {
public:
    static void Read(const NJson::TJsonValue& src, T* dst) {
        dst->Load(src);
    }

    static void Write(const T& value, NJson::TJsonValue* dst) {
        value.Save(dst);
    }
};

template <class T>
class TJsonFieldHelper<T, true> {
public:
    static void Read(const NJson::TJsonValue& src, T* dst) {
        (*dst) = FromString<T>(src.GetStringSafe());
    }

    static void Write(const T& value, NJson::TJsonValue* dst) {
        (*dst) = ToString(value);
    }
};

#define DECLARE_FIELD_HELPER(cppType, type)                               \
    template <>                                                           \
    class TJsonFieldHelper<cppType, false> {                              \
    public:                                                               \
        static void Read(const NJson::TJsonValue& src, cppType* dst) {    \
            (*dst) = src.Get##type##Safe();                               \
        }                                                                 \
                                                                          \
        static void Write(const cppType& value, NJson::TJsonValue* dst) { \
            (*dst) = value;                                               \
        }                                                                 \
    };

DECLARE_FIELD_HELPER(double, Double);

DECLARE_FIELD_HELPER(float, Double);

DECLARE_FIELD_HELPER(ui32, UInteger);

DECLARE_FIELD_HELPER(int, Integer);

DECLARE_FIELD_HELPER(ui64, UInteger);

DECLARE_FIELD_HELPER(i64, Integer);

DECLARE_FIELD_HELPER(TString, String);

DECLARE_FIELD_HELPER(TStringBuf, String);

DECLARE_FIELD_HELPER(bool, Boolean);

#undef DECLARE_FIELD_HELPER

template <class T>
class TJsonFieldHelper<TVector<T>, false> {
public:
    static Y_NO_INLINE void Read(const NJson::TJsonValue& src, TVector<T>* dst) {
        dst->clear();
        if (src.IsArray()) {
            const NJson::TJsonValue::TArray& data = src.GetArraySafe();
            dst->resize(data.size());
            for (ui32 i = 0; i < dst->size(); ++i) {
                TJsonFieldHelper<T>::Read(data.at(i), &(*dst)[i]);
            }
        } else {
            T tmp;
            TJsonFieldHelper<T>::Read(src, &tmp);
            dst->push_back(std::move(tmp));
        }
    }

    static Y_NO_INLINE void Write(const TVector<T>& src, NJson::TJsonValue* dst) {
        (*dst) = NJson::TJsonValue(NJson::EJsonValueType::JSON_ARRAY);
        for (const auto& entry : src) {
            NJson::TJsonValue value;
            TJsonFieldHelper<T>::Write(entry, &value);
            dst->AppendValue(std::move(value));
        }
    }
};

namespace {
    // TMap / THashMap
    template <class TMapping>
    class TJsonFieldHelperImplForMapping {
        using TKey = typename TMapping::key_type;
        using T = typename TMapping::mapped_type;

    public:
        static Y_NO_INLINE void Read(const NJson::TJsonValue& src, TMapping* dst) {
            dst->clear();
            if (src.IsMap()) {
                const auto& data = src.GetMapSafe();
                for (const auto& entry : data) {
                    TJsonFieldHelper<T>::Read(entry.second, &((*dst)[FromString<TKey>(entry.first)]));
                }
            } else {
                ythrow TCatBoostException() << "Error: wrong json type";
            }
        }

        static Y_NO_INLINE void Write(const TMapping& src, NJson::TJsonValue* dst) {
            (*dst) = NJson::TJsonValue(NJson::EJsonValueType::JSON_MAP);
            for (const auto& entry : src) {
                NJson::TJsonValue value;
                TJsonFieldHelper<T>::Write(entry.second, &value);
                (*dst)[ToString<TKey>(entry.first)] = std::move(value);
            }
        }
    };
}

template <class TKey, class T>
class TJsonFieldHelper<TMap<TKey, T>, false> : public TJsonFieldHelperImplForMapping<TMap<TKey, T>> {};

template <class TKey, class T>
class TJsonFieldHelper<THashMap<TKey, T>, false> : public TJsonFieldHelperImplForMapping<THashMap<TKey, T>> {};

template <>
class TJsonFieldHelper<NJson::TJsonValue, false> {
public:
    static void Read(const NJson::TJsonValue& src, NJson::TJsonValue* dst) {
        (*dst) = src;
    }

    static void Write(const NJson::TJsonValue& src, NJson::TJsonValue* dst) {
        (*dst) = src;
    }
};


template <class T>
class TJsonFieldHelper<TMaybe<T>, false> {
public:
    static Y_NO_INLINE void Read(const NJson::TJsonValue& src, TMaybe<T>* dst) {
        if (src.IsNull()) {
            *dst = Nothing();
        } else {
            T value;
            TJsonFieldHelper<T>::Read(src, &value);
            *dst = value;
        }
    }

    static Y_NO_INLINE void Write(const TMaybe<T>& src, NJson::TJsonValue* dst) {
        CB_ENSURE(dst, "Error: can't write to nullptr");
        if (!src) {
            *dst = NJson::TJsonValue(NJson::EJsonValueType::JSON_NULL);
        } else {
            TJsonFieldHelper<T>::Write(src.GetRef(), dst);
        }
    }
};
