#pragma once

#include <util/stream/fwd.h>
#include <util/system/types.h>

namespace NCB {
    struct TQuantizedPool;
}

namespace NCB {
    enum class EQuantizedPoolPrintFormat : ui8 {
        Unknown                 = 0,
        HumanReadableChunkWise  = 1,
        HumanReadableColumnWise = 2
    };

    struct TPrintQuantizedPoolParameters {
        EQuantizedPoolPrintFormat Format = EQuantizedPoolPrintFormat::Unknown;
        bool ResolveBorders = false;
    };

    // Print quantized pool in readable format.
    //
    // For serialization please use `SaveQuantizedPool` or `LoadQuantizedPool`.
    void PrintQuantizedPool(
        const TQuantizedPool& pool,
        const TPrintQuantizedPoolParameters& params,
        IOutputStream* output);
}
