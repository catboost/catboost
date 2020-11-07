#pragma once

#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/polymorphic_type_containers.h>

#include <catboost/private/libs/options/binarization_options.h>
#include <catboost/private/libs/options/enums.h>

#include <library/cpp/grid_creator/binarization.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/system/types.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

#include <type_traits>


struct TRestorableFastRng64;


namespace NCB {

    constexpr int BINARIZATION_BLOCK_SIZE = 16384;

    template <class TBinType>
    inline TBinType GetBinFromBorders(TConstArrayRef<float> borders,
                                      float value) {
        static_assert(std::is_unsigned<TBinType>::value, "TBinType must be an unsigned integer");

        ui32 index = 0;
        if (borders.size() <= 64) {
            for (float border : borders) {
                index += (value > border);
            }
        } else {
            index = LowerBound(borders.begin(), borders.end(), value) - borders.begin();
        }

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

        if (std::isnan(value)) {
            // For ENanMode::Forbidden we choose 0 because that's how it's done on CPU both for
            // training and model application, see:
            //
            // catboost/private/libs/algo/quantization.cpp:128 at r3969212
            return (nanMode == ENanMode::Max) ? borders.size() : 0;
        } else {
            return GetBinFromBorders<TBinType>(borders, value);
        }
    }

    template <typename TQuantizedBin>
    inline TQuantizedBin Quantize(ui32 featureIdx,
                                  bool allowNans,
                                  ENanMode nanMode,
                                  TConstArrayRef<float> borders,
                                  float srcValue) {

        if (std::isnan(srcValue)) {
            CB_ENSURE(
                allowNans,
                "There are NaNs in test dataset (feature number "
                << featureIdx << ") but there were no NaNs in learn dataset"
            );
            return (nanMode == ENanMode::Max) ? borders.size() : 0;
        } else {
            return GetBinFromBorders<TQuantizedBin>(borders, srcValue);
        }
    }

    // TODO(kirillovs): replace with effective block-quantizer from model evaluation code
    template <typename TQuantizedBin>
    void QuantizeBlock(TConstArrayRef<float> srcFeatureData,
                  bool allowNans,
                  ENanMode nanMode,
                  ui32 featureIdx, // for error message

                  // if nanMode != ENanMode::Forbidden borders must include -min_float or +max_float
                  TConstArrayRef<float> borders,
                  TArrayRef<TQuantizedBin> quantizedData) {
        Y_ASSERT(srcFeatureData.size() == quantizedData.size());
        for (size_t i : xrange(srcFeatureData.size())) {
            quantizedData[i] = Quantize<TQuantizedBin>(featureIdx,
                                                         allowNans,
                                                         nanMode,
                                                         borders,
                                                         srcFeatureData[i]);
        }
    }

    template <typename TQuantizedBin>
    void Quantize(const ITypedArraySubset<float>& srcFeatureData,
                  bool allowNans,
                  ENanMode nanMode,
                  ui32 featureIdx, // for error message

                  // if nanMode != ENanMode::Forbidden borders must include -min_float or +max_float
                  TConstArrayRef<float> borders,
                  TArrayRef<TQuantizedBin> quantizedData,
                  NPar::ILocalExecutor* localExecutor) {

        srcFeatureData.ParallelForEach(
            [=] (ui32 idx, float srcValue) {
                quantizedData[idx] = Quantize<TQuantizedBin>(featureIdx,
                                                             allowNans,
                                                             nanMode,
                                                             borders,
                                                             srcValue);
            },
            localExecutor,
            BINARIZATION_BLOCK_SIZE
        );
    }


    inline ui32 GetSampleSizeForBorderSelectionType(ui32 vecSize,
                                                    EBorderSelectionType /*borderSelectionType*/,
                                                    ui32 slowSubsetSize = 100000) {
        return Min<ui32>(vecSize, slowSubsetSize);
    };

    TArraySubsetIndexing<ui32> GetArraySubsetForBuildBorders(ui32 objectCount,
                                                             EBorderSelectionType borderSelectionType,
                                                             bool isRandomShuffled,
                                                             ui32 slowSubsetSize,
                                                             TRestorableFastRng64* rand);

    TVector<float> BuildBorders(TConstArrayRef<float> floatFeature,
                                const ui32 seed,
                                const NCatboostOptions::TBinarizationOptions& config);

    //this routine assumes NanMode = Min/Max means we have nans in float-values
    template <class TBinType = ui32>
    inline TVector<TBinType> BinarizeLine(TConstArrayRef<float> values,
                                          const ENanMode nanMode,
                                          TConstArrayRef<float> borders) {
        static_assert(std::is_unsigned<TBinType>::value, "TBinType must be an unsigned integer");

        TVector<TBinType> result;
        result.yresize(values.size());

        NPar::ILocalExecutor::TExecRangeParams params(0, (int)values.size());
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

    inline ui8 CalcHistogramWidthForBorders(size_t bordersCount) {
        Y_ASSERT(bordersCount < 65536);
        if (bordersCount < 256) {
            return 8;
        }
        return 16;
    }
}
