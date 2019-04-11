#pragma once

#include <catboost/libs/data_util/path_with_scheme.h>

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
    TQuantizedPool LoadQuantizedPool(const TPathWithScheme& pathWithScheme, const TLoadQuantizedPoolParameters& params);

    NIdl::TPoolQuantizationSchema LoadQuantizationSchemaFromPool(TStringBuf path);
    NIdl::TPoolMetainfo LoadPoolMetainfo(TStringBuf path);
    TQuantizedPoolDigest CalculateQuantizedPoolDigest(TStringBuf path);
    void AddPoolMetainfo(const NIdl::TPoolMetainfo& metainfo, TQuantizedPool* const pool);
}
