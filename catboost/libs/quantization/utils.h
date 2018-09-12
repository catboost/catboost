#pragma once

#include <catboost/libs/helpers/exception.h>

#include <catboost/libs/options/binarization_options.h>
#include <catboost/libs/options/enums.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/system/types.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

#include <type_traits>


namespace NCB {

    constexpr int BINARIZATION_BLOCK_SIZE = 16384;

    template <class TBinType>
    inline TBinType GetBinFromBorders(TConstArrayRef<float> borders,
                                      float value) {
        static_assert(std::is_unsigned<TBinType>::value, "TBinType must be an unsigned integer");

        ui32 index = 0;
        while (index < borders.size() && value > borders[index])
            ++index;

        TBinType resultIndex = static_cast<TBinType>(index);

        CB_ENSURE(
            static_cast<ui32>(resultIndex) == index,
            "Error: can't binarize to binType for border count " << borders.size()
        );
        return index;
    }

    template <class TBinType>
    inline TBinType Binarize(const ENanMode nanMode,
                             TConstArrayRef<float> borders,
                             float value) {
        static_assert(std::is_unsigned<TBinType>::value, "TBinType must be an unsigned integer");

        if (IsNan(value)) {
            CB_ENSURE(nanMode != ENanMode::Forbidden, "Error, NaNs for current feature are forbidden. (All NaNs are forbidden or test has NaNs and learn not)");
            return (nanMode == ENanMode::Min) ? 0 : borders.size();
        } else {
            TBinType bin = GetBinFromBorders<TBinType>(borders, value);
            return (nanMode == ENanMode::Min) ? (bin + 1) : bin;
        }
    }

    TVector<float> BuildBorders(const TVector<float>& floatFeature,
                                const ui32 seed,
                                const NCatboostOptions::TBinarizationOptions& config);

    //this routine assumes NanMode = Min/Max means we have nans in float-values
    template <class TBinType = ui32>
    inline TVector<TBinType> BinarizeLine(TConstArrayRef<float> values,
                                          const ENanMode nanMode,
                                          TConstArrayRef<float> borders) {
        static_assert(std::is_unsigned<TBinType>::value, "TBinType must be an unsigned integer");

        TVector<TBinType> result;
        result.yresize(values.Size());

        NPar::TLocalExecutor::TExecRangeParams params(0, (int)values.Size());
        params.SetBlockSize(BINARIZATION_BLOCK_SIZE);

        NPar::LocalExecutor().ExecRange([&](int blockIdx) {
            NPar::LocalExecutor().BlockedLoopBody(params, [&](int i) {
                result[i] = Binarize<TBinType>(nanMode, borders, values[i]);
            })(blockIdx);
        },
                                        0, params.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

        return result;
    }

    inline ui32 GetBinCount(TConstArrayRef<float> borders, ENanMode nanMode) {
        return (ui32)borders.size() + 1 + (nanMode != ENanMode::Forbidden);
    }

}
