#pragma once

#include "exception.h"
#include <library/cpp/json/json_value.h>
#include <library/cpp/json/json_writer.h>

#include <util/stream/str.h>

using namespace NJson;


template <typename T>
static TJsonValue VectorToJson(const TVector<T>& values) {
    TJsonValue jsonValue;
    jsonValue.SetType(EJsonValueType::JSON_ARRAY);
    for (const auto& value: values) {
        jsonValue.AppendValue(TJsonValue(value));
    }
    CB_ENSURE(jsonValue.GetArray().size() == values.size());
    return jsonValue;
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

NJson::TJsonValue ReadTJsonValue(TStringBuf paramsJson);

/*
 * Use this function instead of simple ToString(jsonValue) because it saves floating point values with proper precision
 */
TString WriteTJsonValue(const NJson::TJsonValue& jsonValue);


template <typename T>
static void InsertEnumType(const TString& typeName, const T& value, TJsonValue* jsonValuePtr) {
    jsonValuePtr->InsertValue(typeName, ToString<T>(value));
}

template <typename T>
static void ReadEnumType(const TString& typeName, const TJsonValue& jsonValue, T* valuePtr) {
    *valuePtr = FromString<T>(jsonValue[typeName].GetString());
}
