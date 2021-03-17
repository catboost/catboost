#pragma once

// N-API (node-addon-api)
#include <napi.h>

// Catboost C API
#include <c_api.h>

// Using STD version of string as it is used by N-API
#include <string>

namespace NHelper {

inline void Check(Napi::Env env, bool condition, const std::string& message) {
    if (!condition) {
        Napi::TypeError::New(env, message)
	    .ThrowAsJavaScriptException();
    }
}

template <typename T>
inline void CheckNotNull(Napi::Env env, T* ptr, const std::string& message) {
    Check(env, ptr != nullptr, message);
}

inline void CheckNotNullHandle(Napi::Env env, ModelCalcerHandle* handle) {
    return CheckNotNull(env, handle, "Internal error - null handle encountered");
}

inline void CheckStatus(Napi::Env& env, bool status) {
    if (!status) {
        const char* errorMessage = GetErrorString();
        CheckNotNull(env, errorMessage, "Internal error - error message expected, but missing");
        Napi::Error::New(env, errorMessage).ThrowAsJavaScriptException();
    }
}

// Matrix types in N-API
enum ENApiType {
    NAT_NUMBER,
    NAT_STRING,
};

bool IsMatrix(const Napi::Value& value, ENApiType type);

}