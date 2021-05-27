#pragma once

#include <util/generic/vector.h>
#include <util/system/types.h>


namespace NCB {
    struct TTagDescription {
        TVector<ui32> Features;

    public:
        bool operator==(const TTagDescription& rhs) const {
            return Features == rhs.Features;
        }
    };
}
