#include "histograms_helper.h"
#include <util/system/env.h>

bool IsReduceCompressed() {
    static const bool reduceCompressed = GetEnv("CB_COMPRESSED_REDUCE", "false") == "true";
    return reduceCompressed;
}
