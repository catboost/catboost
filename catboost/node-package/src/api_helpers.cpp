#include "api_helpers.h"

#include <util/system/yassert.h>

namespace NHelper {

bool IsMatrix(const Napi::Value& value, ENApiType type) {
    if (!value.IsArray()) {
        return false;
    }
    const Napi::Array floatFeatures = value.As<Napi::Array>();
    const uint32_t rowsCount = floatFeatures.Length();
    if (rowsCount == 0) {
        return true;
    }

    if (!floatFeatures[0u].IsArray()) {
        return false;
    }
    const uint32_t columnsCount = floatFeatures[0u].As<Napi::Array>().Length();

    for (uint32_t i = 0; i < rowsCount; ++i) {
        if (!floatFeatures[i].IsArray()) {
            return false;
        }

        const Napi::Array row = floatFeatures[i].As<Napi::Array>();
        if (row.Length() != columnsCount) {
            return false;
        }

        for (uint32_t j = 0; j < rowsCount; ++j) {
            switch (type) {
                case ENApiType::NAT_NUMBER:
                    if (!row[j].IsNumber()) {
                        return false;
                    }
                    break;
                case ENApiType::NAT_STRING:
                    if (!row[j].IsString()) {
                        return false;
                    }
                    break;
                default:
                    Y_ASSERT(false);
            }
        }
    }

    return true;
}

}
