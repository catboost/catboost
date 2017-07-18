#pragma once

#include "last_getopt_support.h"

#include <util/string/split.h>

namespace NLastGetopt {
    /// Handler to split option value by delimiter into a target container.
    template <class Container>
    struct TOptSplitHandler;

    /// Handler to split option value by delimiter into a target container and allow ranges.
    template <class Container>
    struct TOptRangeSplitHandler;

    /// Handler to parse key-value pairs (default delimiter is '=') and apply user-supplied handler to each pair
    template <class TpFunc>
    struct TOptKVHandler;

    void PrintUsageAndExit(const TOptsParser* parser);
    void PrintVersionAndExit(const TOptsParser* parser);
}
