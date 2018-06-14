#pragma once

#include <util/stream/fwd.h>
#include <util/system/types.h>

namespace NCB {
    struct TQuantizedPool;
}

namespace NCB {
    enum class EQuantizedPoolPrintFormat : ui8 {
        Unknown                     = 0,
        HumanReadable               = 1,
        HumanReadableResolveBorders = 2
    };

    struct TPrintQuantizedPoolParameters {
        EQuantizedPoolPrintFormat Format = EQuantizedPoolPrintFormat::Unknown;
    };

    // Print quantized pool in readable format.
    //
    // For serialization please use `SaveQuantizedPool` or `LoadQuantizedPool`.
    void PrintQuantizedPool(
        const TQuantizedPool& pool,
        const TPrintQuantizedPoolParameters& params,
        IOutputStream* output);
}
