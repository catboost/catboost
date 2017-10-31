#pragma once

#include "numeric.h"
#include <util/generic/hash.h>
#include <util/generic/array_ref.h>

template <typename ElementHash = void>
class TRangeHash {
private:
    template <typename ElementType>
    using TBase = std::conditional_t<
        !std::is_void<ElementHash>::value,
        ElementHash,
        THash<ElementType>>;

public:
    template <typename Range>
    size_t operator()(const Range& range) const {
        size_t accumulated = 0;
        for (const auto& element : range) {
            accumulated = CombineHashes(accumulated, TBase<typename Range::value_type>()(element));
        }
        return accumulated;
    }
};

using TSimpleRangeHash = TRangeHash<>;

template <typename RegionHash = void>
class TContiguousHash {
private:
    template <typename ElementType>
    using TBase = std::conditional_t<
        !std::is_void<RegionHash>::value,
        RegionHash,
        TRangeHash<ElementType>>;

public:
    template <typename ContainerType>
    auto operator()(const ContainerType& container) const {
        return operator()(MakeArrayRef(container));
    }

    template <typename ElementType>
    auto operator()(const TArrayRef<ElementType>& data) const {
        return TBase<ElementType>()(data);
    }
};
