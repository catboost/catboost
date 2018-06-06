#pragma once

#include "pool.h"

#include <util/stream/fwd.h>
#include <util/system/types.h>

namespace NCB {
    enum class EQuantizedPoolPrintFormat : ui8 {
        Unknown       = 0,
        HumanReadable = 1
    };

    struct TPrintQuantizationSchemaParameters {
        EQuantizedPoolPrintFormat Format{EQuantizedPoolPrintFormat::Unknown};
    };

    // Print quantized pool in readable format.
    //
    // For serialization please use `SaveQuantizedPool` or `LoadQuantizedPool`.
    void PrintQuantizedPool(
        const TQuantizedPool& schema,
        const TPrintQuantizationSchemaParameters& params,
        IOutputStream* output);
}
