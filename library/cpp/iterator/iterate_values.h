#pragma once

#include "mapped.h"

template<typename TMapping>
auto IterateValues(TMapping&& map) {
    return ::MakeMappedRange(
        std::forward<TMapping>(map),
        [](auto& x) -> decltype((x.second)) {
            return x.second;
        }
    );
}
