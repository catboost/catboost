#pragma once

#include <util/generic/vector.h>
#include <util/system/types.h>


namespace NCB {
    struct TTagDescription {
        TVector<ui32> Features;
        float Cost = 1.0;

    public:
        TTagDescription()
            : Features({})
        { }

        TTagDescription(const TVector<ui32>& features, float cost = 1.0)
            : Features(features)
            , Cost(cost)
        {}

        bool operator==(const TTagDescription& rhs) const {
            return Features == rhs.Features && Cost == rhs.Cost;
        }
    };
}
