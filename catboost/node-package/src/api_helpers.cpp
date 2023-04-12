#include "api_helpers.h"

#include <assert.h>
#include <string>

namespace NHelper {

bool CheckIsMatrix(Napi::Env env, const Napi::Value& value, ENApiType type, const std::string& errorPrefix) {
    if (!Check(env, value.IsArray(), errorPrefix + "is not an array")) {
        return false;
    }
    const Napi::Array floatFeatures = value.As<Napi::Array>();
    const uint32_t rowsCount = floatFeatures.Length();
    if (rowsCount == 0) {
        return true;
    }

    if (!Check(env, floatFeatures[0u].IsArray(), errorPrefix + "first element of matrixn is not an array")) {
        return false;
    }
    const uint32_t columnsCount = floatFeatures[0u].As<Napi::Array>().Length();
    size_t numberCount = 0;
    size_t strCount = 0;
    for (uint32_t i = 0; i < rowsCount; ++i) {
        if (!Check(
            env,
            floatFeatures[i].IsArray(),
            errorPrefix + std::to_string(i) + "-th element of matrix is not an array")
        ) {
            return false;
        }

        const Napi::Array row = floatFeatures[i].As<Napi::Array>();
        if (!Check(
            env,
            row.Length() == columnsCount,
            errorPrefix + "invalid length of " + std::to_string(i) + "-th row")
        ) {
            return false;
        }

        for (uint32_t j = 0; j < columnsCount; ++j) {
            if (row[j].IsNumber()) {
                ++numberCount;
            } else if (row[j].IsString()) {
                ++strCount;
            } else {
                std::string typeStr = "";
                switch(row[j].Type()) {
                    case napi_undefined:
                        typeStr = "napi_undefined";
                        break;
                    case napi_null:
                        typeStr = "napi_null";
                        break;
                    case napi_boolean:
                        typeStr = "napi_boolean";
                        break;
                    case napi_number:
                        typeStr = "napi_number";
                        break;
                    case napi_string:
                        typeStr = "napi_string";
                        break;
                    case napi_symbol:
                        typeStr = "napi_symbol";
                        break;
                    case napi_object:
                        typeStr = "napi_object";
                        break;
                    case napi_function:
                        typeStr = "napi_function";
                        break;
                    case napi_external:
                        typeStr = "napi_external";
                        break;
                    //case napi_bigint:
                    //   typeStr = "napi_bigint";
                    //    break;
                    default:
                        typeStr = std::to_string(row[j].Type());
                        break;
                }
                Check(env, false, errorPrefix + "invalid type found: " + typeStr);
                return false;
            }
        }
        switch (type) {
            case ENApiType::NAT_NUMBER:
                if (!Check(env, strCount == 0, errorPrefix + "no strings in numeric matrix expected")) {
                    return false;
                }
                break;
            case ENApiType::NAT_STRING:
                if (!Check(env, numberCount == 0, errorPrefix + "no strings in numeric matrix expected")) {
                    return false;
                }
                break;
            case ENApiType::NAT_NUMBER_OR_STRING:
                if (!Check(env, !(numberCount > 0 && strCount > 0), errorPrefix + "mixed strings an numbers in array")) {
                    return false;
                }
                break;
            default:
                assert(false);
        }
    }

    return true;
}

}
