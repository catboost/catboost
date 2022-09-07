#pragma once

#include <library/cpp/json/json_writer.h>

#include <util/stream/str.h>

using namespace NJson;


template <typename T>
static TJsonValue VectorToJson(const TVector<T>& values) {
    TJsonValue jsonValue;
    for (const auto& value: values) {
        jsonValue.AppendValue(value);
    }
    return jsonValue;
}

inline void FromJson(const TJsonValue& value, TString* result) {
    *result = value.GetString();
}

static void WriteJsonWithCatBoostPrecision(const TJsonValue& value, bool formatOutput, IOutputStream* out) {
    TJsonWriterConfig config;
    config.FormatOutput = formatOutput;
    config.FloatNDigits = 9;
    config.DoubleNDigits = 17;
    config.SortKeys = true;
    WriteJson(out, &value, config);
}

inline TString WriteJsonWithCatBoostPrecision(const TJsonValue& value, bool formatOutput) {
    TStringStream ss;
    WriteJsonWithCatBoostPrecision(value, formatOutput, &ss);
    return ss.Str();
}

template <typename T>
static void FromJson(const TJsonValue& value, T* result) {
    switch (value.GetType()) {
        case EJsonValueType::JSON_INTEGER:
            *result = T(value.GetInteger());
            break;
        case EJsonValueType::JSON_DOUBLE:
            *result = T(value.GetDouble());
            break;
        case EJsonValueType::JSON_UINTEGER:
            *result = T(value.GetUInteger());
            break;
        default:
            Y_ASSERT(false);
    }
}

template <typename T>
static T FromJson(const TJsonValue& value) {
    T result;
    FromJson(value, &result);
    return result;
}

template <typename T>
static TVector<T> JsonToVector(const TJsonValue& jsonValue) {
    TVector<T> result;
    for (const auto& value: jsonValue.GetArray()) {
        result.push_back(FromJson<T>(value));
    }
    return result;
}

template <typename T>
static void InsertEnumType(const TString& typeName, const T& value, TJsonValue* jsonValuePtr) {
    jsonValuePtr->InsertValue(typeName, ToString<T>(value));
}
