#include "api_helpers.h"

#include <assert.h>

namespace NHelper {

bool CheckIsMatrix(Napi::Env env, const Napi::Value& value, ENApiType type, const std::string& errorPrefix) {
    if (!Check(env, value.IsArray(), errorPrefix + "is not an array")) {
        return false;
    }
    const Napi::Array matrix = value.As<Napi::Array>();
    const uint32_t rowsCount = matrix.Length();
    if (rowsCount == 0) {
        return true;
    }

    if (!Check(env, matrix[0u].IsArray(), errorPrefix + "the first element of the matrix is not an array")) {
        return false;
    }
    const uint32_t columnsCount = matrix[0u].As<Napi::Array>().Length();
    size_t numberCount = 0;
    size_t strCount = 0;

    std::function<bool(const Napi::Value&)> checkElement;

    switch (type) {
        case ENApiType::NAT_NUMBER:
            checkElement = [&] (const Napi::Value& value) -> bool {
                return Check(
                    env,
                    value.IsNumber(),
                    "non-numeric type in the matrix elements"
                );
            };
            break;
        case ENApiType::NAT_STRING:
            checkElement = [&] (const Napi::Value& value) -> bool {
                return Check(
                    env,
                    value.IsString(),
                    "non-string type in the matrix elements"
                );
            };
            break;
        case ENApiType::NAT_NUMBER_OR_STRING:
            checkElement = [&] (const Napi::Value& value) -> bool {
                if (value.IsNumber()) {
                    ++numberCount;
                } else if (value.IsString()) {
                    ++strCount;
                } else {
                    Check(env, false, errorPrefix + "invalid type found: " + std::to_string(value.Type()));
                    return false;
                }
                return true;
            };
            break;
        case ENApiType::NAT_ARRAY_OR_NUMBERS:
            checkElement = [&] (const Napi::Value& value) -> bool {
                if (!Check(
                        env,
                        value.IsArray(),
                        "the matrix contains non-array elements"
                    ))
                {
                    return false;
                }

                const Napi::Array subArray = value.As<Napi::Array>();
                const uint32_t subArraySize = subArray.Length();

                for (uint32_t k = 0; k < subArraySize; ++k) {
                    if (!Check(
                            env,
                            subArray[k].IsNumber(),
                            "an array in the matrix element contains a non-number element"
                        ))
                    {
                        return false;
                    }
                }
                return true;
            };
            break;
    }


    for (uint32_t i = 0; i < rowsCount; ++i) {
        if (!Check(
                env,
                matrix[i].IsArray(),
                errorPrefix + std::to_string(i) + "-th element of the matrix is not an array"
            ))
        {
            return false;
        }

        const Napi::Array row = matrix[i].As<Napi::Array>();
        if (!Check(
                env,
                row.Length() == columnsCount,
                errorPrefix + "invalid length of " + std::to_string(i) + "-th row"
            ))
        {
            return false;
        }

        for (uint32_t j = 0; j < columnsCount; ++j) {
            if (!checkElement(row[j])) {
                return false;
            }
        }

        if (type == ENApiType::NAT_NUMBER_OR_STRING) {
            if (!Check(
                    env,
                    !(numberCount > 0 && strCount > 0),
                    errorPrefix + "mixed strings and numbers in the matrix"
                ))
            {
                return false;
            }
        }
    }

    return true;
}

}
