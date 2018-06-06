#pragma once

#include <util/generic/fwd.h>

namespace NCB {
    struct TQuantizedPool;
}

namespace NCB {
    TVector<TString> SaveQuantizedPool(
        const TQuantizedPool& schema,
        TStringBuf directory,
        TStringBuf basename,
        TStringBuf extension);

    struct TLoadQuantizedPoolParameters {
        bool LockMemory{true};
        bool Precharge{true};
    };

    // Load quantized pool saved by `SaveQuantizedPool` from files.
    TQuantizedPool LoadQuantizedPool(
        TConstArrayRef<TString> files,
        const TLoadQuantizedPoolParameters& params);
}
