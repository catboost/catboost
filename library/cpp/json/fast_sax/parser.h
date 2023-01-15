#pragma once

#include <library/cpp/json/common/defs.h>

namespace NJson {
    bool ReadJsonFast(TStringBuf in, TJsonCallbacks* callbacks);

    inline bool ValidateJsonFast(TStringBuf in, bool throwOnError = false) {
        Y_ASSERT(false); // this method is broken, see details in IGNIETFERRO-1243. Use NJson::ValidateJson instead, or fix this one before using
        TJsonCallbacks c(throwOnError);
        return ReadJsonFast(in, &c);
    }
}
