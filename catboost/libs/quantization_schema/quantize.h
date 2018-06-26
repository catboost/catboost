#pragma once

#include <catboost/libs/options/enums.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/ymath.h>

namespace NCB {
    inline size_t Quantize(const float value, const TConstArrayRef<float> borders, const ENanMode nanMode) {
        if (IsNan(value)) {
            CB_ENSURE(ENanMode::Forbidden != nanMode, "NaNs are forbidden");
            return ENanMode::Min == nanMode ? 0 : borders.size();
        }

        if (borders.size() <= 50) {
            size_t index = 0;
            while (index < borders.size() && borders[index] < value) {
                ++index;
            }
            return index;
        }

        return LowerBound(borders.begin(), borders.end(), value) - borders.begin();
    }
}
