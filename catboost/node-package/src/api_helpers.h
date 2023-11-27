#pragma once

// N-API (node-addon-api)
#include <napi.h>

// Catboost C API
#include <catboost/libs/model_interface/c_api.h>

#include <vector>

// used by N-API.
#include <string>

namespace NHelper {

// Checks if the condition is true, schedules JS exception otherwise.
// Returns false if check failed.
inline bool Check(Napi::Env env, bool condition, const std::string& message) {
    if (!condition) {
        Napi::TypeError::New(env, message).ThrowAsJavaScriptException();
    }

    return condition;
}

// Checks if the pointer is not null, throws JS exception otherwise.
template <typename T>
inline bool CheckNotNull(Napi::Env env, T* ptr, const std::string& message) {
    return Check(env, ptr != nullptr, message);
}

// Checks that the model handle is not null. As this should never be the case throws internal error.
inline bool CheckNotNullHandle(Napi::Env env, ModelCalcerHandle* handle) {
    return CheckNotNull(env, handle, "Internal error - null handle encountered");
}

// Checks that the return status of C API is true, returns error in JS exception otherwise.
inline bool CheckStatus(Napi::Env& env, bool status) {
    if (!status) {
        const char* errorMessage = GetErrorString();
        CheckNotNull(env, errorMessage, "Internal error - error message expected, but missing");
        Napi::Error::New(env, errorMessage).ThrowAsJavaScriptException();
    }

    return status;
}

// Matrix types in N-API
enum ENApiType {
    NAT_NUMBER,
    NAT_STRING,
    NAT_NUMBER_OR_STRING,
    NAT_ARRAY_OR_NUMBERS
};

// Checks if the value a matrix with element of a given type.
bool CheckIsMatrix(Napi::Env env, const Napi::Value& value, ENApiType type, const std::string& errorStr);

// Converts vector of numbers to N-API array.
template <typename T>
Napi::Array ConvertToArray(Napi::Env env, const std::vector<T>& values) {
    Napi::Array result = Napi::Array::New(env);
    uint32_t index = 0;
    for (const auto value: values) {
        result[index++] = Napi::Number::New(env, static_cast<double>(value));
    }

    return result;
}

}
