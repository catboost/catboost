#pragma once

#include <util/generic/fwd.h>
#include <util/stream/fwd.h>

namespace NCB {
    struct TQuantizedPool;
    struct TQuantizedPoolDigest;

    namespace NIdl {
        class TPoolMetainfo;
        class TPoolQuantizationSchema;
    }
}

namespace NCB {
    void SaveQuantizedPool(const TQuantizedPool& pool, IOutputStream* output);

    struct TLoadQuantizedPoolParameters {
        bool LockMemory = true;
        bool Precharge = true;
    };

    // Load quantized pool saved by `SaveQuantizedPool` from file.
    TQuantizedPool LoadQuantizedPool(TStringBuf path, const TLoadQuantizedPoolParameters& params);

    // TODO(yazevnul): rename it to `LoadQuantizationSchemaFromPool`
    NIdl::TPoolQuantizationSchema LoadQuantizationSchema(TStringBuf path);
    NIdl::TPoolMetainfo LoadPoolMetainfo(TStringBuf path);
    TQuantizedPoolDigest CalculateQuantizedPoolDigest(TStringBuf path);
}
