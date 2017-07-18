#pragma once

#include <library/json/common/defs.h>

namespace NJson {
    bool ReadJsonFast(TStringBuf in, TJsonCallbacks* callbacks);

    inline bool ValidateJsonFast(TStringBuf in, bool throwOnError = false) {
        TJsonCallbacks c(throwOnError);
        return ReadJsonFast(in, &c);
    }
}
