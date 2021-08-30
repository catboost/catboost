#pragma once

#include <stddef.h>

namespace NHnsw {
    template <class T, class TDistance>
    struct TDistanceWithDimension : TDistance {
        const size_t Dimension = 0;
        TDistanceWithDimension() = default;
        TDistanceWithDimension(const TDistance& base, size_t dimension)
            : TDistance(base)
            , Dimension(dimension)
        {
        }
        auto operator()(const T* a, const T* b) const {
            return TDistance::operator()(a, b, Dimension);
        }
    };

}
