#pragma once

#include "numeric.h"
#include <util/str_stl.h>

template <typename TOne>
constexpr size_t MultiHash(const TOne& one) noexcept {
    return THash<TOne>()(one);
}

template <typename THead, typename... TTail>
constexpr size_t MultiHash(const THead& head, const TTail&... tail) noexcept {
    return CombineHashes(MultiHash(tail...), THash<THead>()(head));
}
