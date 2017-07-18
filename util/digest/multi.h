#pragma once

#include <util/str_stl.h>

#include "numeric.h"

template <typename TOne>
size_t MultiHash(const TOne& one) noexcept {
    return THash<TOne>()(one);
}
template <typename THead, typename... TTail>
size_t MultiHash(const THead& head, TTail&&... tail) noexcept {
    return CombineHashes(MultiHash(tail...), THash<THead>()(head));
}
