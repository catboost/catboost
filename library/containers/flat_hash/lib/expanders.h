#pragma once

#include <utility>

namespace NFlatHash {

struct TSimpleExpander {
    static constexpr bool NeedGrow(size_t size, size_t buckets) noexcept {
        return size >= buckets / 2;
    }

    static constexpr bool WillNeedGrow(size_t size, size_t buckets) noexcept {
        return NeedGrow(size + 1, buckets);
    }

    static constexpr size_t EvalNewSize(size_t buckets) noexcept {
        return buckets * 2;
    }

    static constexpr size_t SuitableSize(size_t size) noexcept {
        return size * 2 + 1;
    }
};

}  // namespace NFlatHash
