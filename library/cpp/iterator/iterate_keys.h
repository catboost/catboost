#pragma once

#include "mapped.h"

template<typename TMapping>
auto IterateKeys(TMapping&& map) {
    return ::MakeMappedRange(
        std::forward<TMapping>(map),
        [](const auto& x) -> decltype((x.first)) {
            return x.first;
        }
    );
}
