#pragma once

#include <array>
#include <catboost/libs/helpers/exception.h>
#include <util/ysaveload.h>
#include <util/generic/vector.h>

namespace NCatboostCuda {
    struct TFoldsHistogram {
        std::array<ui32, 9> Counts;

        TFoldsHistogram() {
            Counts.fill(0);
        }

        ui32 FeatureCountForBits(ui32 fromBit, ui32 toBitInclusive) const {
            CB_ENSURE(toBitInclusive <= 8);
            CB_ENSURE(fromBit <= toBitInclusive);
            ui32 count = 0;
            for (ui32 bit = fromBit; bit <= toBitInclusive; ++bit) {
                count += Counts[bit];
            }
            return count;
        }

        Y_SAVELOAD_DEFINE(Counts);
    };
}
