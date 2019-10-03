#pragma once

#include <catboost/private/libs/options/enums.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/ymath.h>

namespace NCB {
    inline size_t Quantize(const float value, const TConstArrayRef<float> borders, const ENanMode nanMode) {
        if (IsNan(value)) {
            // Before r548266 NaNs were forbidden if `nanMode==ENanMode::Forbidden`, but it was
            // decided that this feature is too annoing, and we should allays allow NaNs (especially
            // in case when learn doesn't have them while test set has).
            //
            // see MLTOOLS-2235
            //
            // For ENanMode::Forbidden we chose bucket 0 because it should always exist.
            return ENanMode::Max == nanMode ? borders.size() : 0;
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
