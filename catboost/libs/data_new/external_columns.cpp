#include "external_columns.h"

#include <catboost/libs/helpers/double_array_iterator.h>
#include <catboost/libs/quantization/utils.h>

#include <util/generic/algorithm.h>
#include <util/generic/utility.h>
#include <util/generic/ylimits.h>
#include <util/system/compiler.h>
#include <util/system/yassert.h>


namespace NCB {

    THolder<ICloneableQuantizedFloatValuesHolder> TExternalFloatValuesHolder::CloneWithNewSubsetIndexing(
        const TFeaturesArraySubsetIndexing* subsetIndexing
    ) const {
        return MakeHolder<TExternalFloatValuesHolder>(
            GetId(),
            SrcData,
            subsetIndexing,
            QuantizedFeaturesInfo
        );
    }

    NCB::TMaybeOwningArrayHolder<ui8> TExternalFloatValuesHolder::ExtractValues(
        NPar::TLocalExecutor* localExecutor
    ) const {
        TVector<ui8> result;
        result.yresize(GetSize());

        const auto floatFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Float>(*this);
        const auto nanMode = QuantizedFeaturesInfo->GetNanMode(floatFeatureIdx);

        // it's ok even if it is learn data, for learn nans are checked at CalcBordersAndNanMode stage
        bool allowNans = (nanMode != ENanMode::Forbidden) ||
            QuantizedFeaturesInfo->GetFloatFeaturesAllowNansInTestOnly();

        Quantize(
            TMaybeOwningConstArraySubset<float, ui32>(&SrcData, SubsetIndexing),
            allowNans,
            nanMode,
            GetId(),
            QuantizedFeaturesInfo->GetBorders(floatFeatureIdx),
            MakeArrayRef(result),
            localExecutor
        );

        return NCB::TMaybeOwningArrayHolder<ui8>::CreateOwning(std::move(result));
    }


    THolder<ICloneableQuantizedCatValuesHolder> TExternalCatValuesHolder::CloneWithNewSubsetIndexing(
        const TFeaturesArraySubsetIndexing* subsetIndexing
    ) const {
        return MakeHolder<TExternalCatValuesHolder>(
            GetId(),
            SrcData,
            subsetIndexing,
            QuantizedFeaturesInfo
        );
    }

    NCB::TMaybeOwningArrayHolder<ui32> TExternalCatValuesHolder::ExtractValues(
        NPar::TLocalExecutor* localExecutor
    ) const {
        TVector<ui32> result;
        result.yresize(GetSize());

        const auto catFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Categorical>(
            *this
        );
        const auto& perfectHash = QuantizedFeaturesInfo->GetCategoricalFeaturesPerfectHash(catFeatureIdx);

        TMaybeOwningConstArraySubset<ui32, ui32>(&SrcData, SubsetIndexing).ParallelForEach(
            [&] (ui32 idx, ui32 srcValue) {
                auto it = perfectHash.find(srcValue); // find is guaranteed to be thread-safe
                Y_ASSERT(it != perfectHash.end());
                result[idx] = it->second.Value;
            },
            localExecutor,
            BINARIZATION_BLOCK_SIZE
        );

        return NCB::TMaybeOwningArrayHolder<ui32>::CreateOwning(std::move(result));
    }

    NCB::TMaybeOwningArrayHolder<ui8> TExternalFloatSparseValuesHolder::ExtractValues(
        NPar::TLocalExecutor* localExecutor
    ) const {
        Y_UNUSED(localExecutor);

        const auto flatFeatureIdx = GetId();
        const auto floatFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Float>(*this);
        const auto nanMode = QuantizedFeaturesInfo->GetNanMode(floatFeatureIdx);

        // it's ok even if it is learn data, for learn nans are checked at CalcBordersAndNanMode stage
        bool allowNans = (nanMode != ENanMode::Forbidden) ||
            QuantizedFeaturesInfo->GetFloatFeaturesAllowNansInTestOnly();

        TConstArrayRef<float> borders = QuantizedFeaturesInfo->GetBorders(floatFeatureIdx);

        const ui8 quantizedDefaultValue
            = Quantize<ui8>(flatFeatureIdx, allowNans, nanMode, borders, SrcData.GetDefaultValue());

        TVector<ui8> result(GetSize(), quantizedDefaultValue);

        TArrayRef<ui8> resultRef = result;

        SrcData.ForEachNonDefault(
            [=] (ui32 nonDefaultIdx, float srcValue) {
                resultRef[nonDefaultIdx]
                    = Quantize<ui8>(flatFeatureIdx, allowNans, nanMode, borders, srcValue);
            }
        );

        return NCB::TMaybeOwningArrayHolder<ui8>::CreateOwning(std::move(result));
    }

    template <class TDst>
    static ui64 EstimateCpuRamLimitForCreateQuantizedSparseSubset(
        const TFeaturesArraySubsetInvertedIndexing& subsetInvertedIndexing,
        ui32 nonDefaultValuesCount,
        ESparseArrayIndexingType sparseArrayIndexingType,
        ui32 dstBitsPerKey
    ) {
        if (HoldsAlternative<TFullSubset<ui32>>(subsetInvertedIndexing)) {
            return 0; // just clone
        }

        ui64 ramUsedForDstIndexing;
        switch (sparseArrayIndexingType) {
            case ESparseArrayIndexingType::Indices:
                ramUsedForDstIndexing = sizeof(ui32) * nonDefaultValuesCount;
                break;
            case ESparseArrayIndexingType::Blocks:
                ramUsedForDstIndexing = 2 * sizeof(ui32) * nonDefaultValuesCount;
                break;
            case ESparseArrayIndexingType::HybridIndex:
                ramUsedForDstIndexing = (sizeof(ui32) + sizeof(ui64)) * nonDefaultValuesCount;
                break;
            default:
                Y_UNREACHABLE();
        }

        const ui64 ramUsedForDstValues = sizeof(TDst) * nonDefaultValuesCount;

        ui64 ramUsedDuringBuilding = ramUsedForDstIndexing + ramUsedForDstValues;
        if (sparseArrayIndexingType != ESparseArrayIndexingType::Indices) {
            // for dstVectorIndexing
            ramUsedDuringBuilding += sizeof(ui32) * nonDefaultValuesCount;
        }

        const TIndexHelper<ui64> indexHelper(dstBitsPerKey);

        ui64 ramUsedDuringSparseCompressedValuesHolderImplCreation
            = ramUsedForDstIndexing + ramUsedForDstValues
                + indexHelper.CompressedSize(nonDefaultValuesCount) * sizeof(ui64);

        return Max(ramUsedDuringBuilding, ramUsedDuringSparseCompressedValuesHolderImplCreation);
    }


    template <class TDst, EFeatureValuesType DstFeatureValuesType, class TSrc, class TQuantizeValueFunction>
    static THolder<TTypedFeatureValuesHolder<TDst, DstFeatureValuesType>> CreateQuantizedSparseSubset(
        ui32 featureId,
        const TSparseArray<TSrc, ui32>& srcData,
        const TInvertedIndexedSubset<ui32>& invertedIndexedSubset,
        TQuantizeValueFunction&& quantizeValueFunction,
        ui32 bitsPerKey
    ) {
        auto srcIndexing = srcData.GetIndexing();
        TConstArrayRef<TSrc> srcNonDefaultValues = *srcData.GetNonDefaultValues();

        TConstArrayRef<ui32> invertedIndicesArray = invertedIndexedSubset.GetMapping();

        TVector<ui32> dstVectorIndexing;
        TVector<TDst> dstValues;

        ui32 nonDefaultValuesIdx = 0;
        srcIndexing->ForEachNonDefault(
            [&](ui32 srcIdx) {
                auto dstIdx = invertedIndicesArray[srcIdx];
                if (dstIdx != TInvertedIndexedSubset<ui32>::NOT_PRESENT) {
                    dstVectorIndexing.push_back(dstIdx);
                    dstValues.push_back(quantizeValueFunction(srcNonDefaultValues[nonDefaultValuesIdx]));
                }
                ++nonDefaultValuesIdx;
            }
        );

        std::function<TCompressedArray(TVector<TDst>&&)> createNonDefaultValuesContainer
            = [&] (TVector<TDst>&& dstValues) {
                return TCompressedArray(
                    dstValues.size(),
                    bitsPerKey,
                    CompressVector<ui64>(dstValues, bitsPerKey)
                );
            };

        return MakeHolder<TSparseCompressedValuesHolderImpl<TDst, DstFeatureValuesType>>(
            featureId,
            MakeSparseArrayBase<TDst, TCompressedArray, ui32>(
                invertedIndexedSubset.GetSize(),
                std::move(dstVectorIndexing),
                std::move(dstValues),
                std::move(createNonDefaultValuesContainer),
                /*sparseArrayIndexingType*/ srcIndexing->GetType(),
                /*ordered*/ false,
                quantizeValueFunction(srcData.GetDefaultValue())
            )
        );
    }

    void TExternalFloatSparseValuesHolder::ScheduleGetSubset(
        const TFeaturesArraySubsetInvertedIndexing* subsetInvertedIndexing,
        TResourceConstrainedExecutor* resourceConstrainedExecutor,
        THolder<IQuantizedFloatValuesHolder>* subsetDst
    ) const {
        const auto floatFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Float>(
            *this
        );

        resourceConstrainedExecutor->Add(
            {
                EstimateCpuRamLimitForCreateQuantizedSparseSubset<ui8>(
                    *subsetInvertedIndexing,
                    SrcData.GetNonDefaultSize(),
                    SrcData.GetIndexing()->GetType(),
                    CalcHistogramWidthForBorders(QuantizedFeaturesInfo->GetBorders(floatFeatureIdx).size())
                ),
                [=] () {
                    if (HoldsAlternative<TFullSubset<ui32>>(*subsetInvertedIndexing)) {
                        // just clone
                        *subsetDst = MakeHolder<TExternalFloatSparseValuesHolder>(
                            this->GetId(),
                            SrcData,
                            QuantizedFeaturesInfo
                        );
                    } else {
                        const ui32 flatFeatureIdx = this->GetId();

                        const auto nanMode = QuantizedFeaturesInfo->GetNanMode(floatFeatureIdx);

                        /* it's ok even if it is learn data, for learn nans are checked at
                         * CalcBordersAndNanMode stage
                         */
                        const bool allowNans = (nanMode != ENanMode::Forbidden) ||
                            QuantizedFeaturesInfo->GetFloatFeaturesAllowNansInTestOnly();

                        TConstArrayRef<float> borders = QuantizedFeaturesInfo->GetBorders(floatFeatureIdx);

                        *subsetDst = CreateQuantizedSparseSubset<ui8, EFeatureValuesType::QuantizedFloat>(
                            this->GetId(),
                            this->SrcData,
                            Get<TInvertedIndexedSubset<ui32>>(*subsetInvertedIndexing),
                            [=] (float srcValue) -> ui8 {
                                return Quantize<ui8>(flatFeatureIdx, allowNans, nanMode, borders, srcValue);
                            },

                            // TODO(akhropov): fix wide histograms support - MLTOOLS-3758
                            sizeof(ui8) * CHAR_BIT
                        );
                    }
                }
            }
        );
    }

    NCB::TMaybeOwningArrayHolder<ui32> TExternalCatSparseValuesHolder::ExtractValues(
        NPar::TLocalExecutor* localExecutor
    ) const {
        Y_UNUSED(localExecutor);

        const auto catFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Categorical>(
            *this
        );
        const auto& perfectHash = QuantizedFeaturesInfo->GetCategoricalFeaturesPerfectHash(catFeatureIdx);

        auto defaultValueIt = perfectHash.find(SrcData.GetDefaultValue());
        Y_ASSERT(defaultValueIt != perfectHash.end());
        const ui32 defaultPerfectHashValue = defaultValueIt->second.Value;

        TVector<ui32> result(GetSize(), defaultPerfectHashValue);

        TArrayRef<ui32> resultRef = result;

        SrcData.ForEachNonDefault(
            [=, &perfectHash] (ui32 nonDefaultIdx, ui32 srcValue) {
                auto it = perfectHash.find(srcValue); // find is guaranteed to be thread-safe
                Y_ASSERT(it != perfectHash.end());
                resultRef[nonDefaultIdx] = it->second.Value;
            }
        );

        return NCB::TMaybeOwningArrayHolder<ui32>::CreateOwning(std::move(result));
    }

    void TExternalCatSparseValuesHolder::ScheduleGetSubset(
        const TFeaturesArraySubsetInvertedIndexing* subsetInvertedIndexing,
        TResourceConstrainedExecutor* resourceConstrainedExecutor,
        THolder<IQuantizedCatValuesHolder>* subsetDst
    ) const {
        const auto catFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Categorical>(
            *this
        );

        resourceConstrainedExecutor->Add(
            {
                EstimateCpuRamLimitForCreateQuantizedSparseSubset<ui32>(
                    *subsetInvertedIndexing,
                    SrcData.GetNonDefaultSize(),
                    SrcData.GetIndexing()->GetType(),
                    sizeof(ui32) * CHAR_BIT
                ),
                [=] () {
                    if (HoldsAlternative<TFullSubset<ui32>>(*subsetInvertedIndexing)) {
                        // just clone
                        *subsetDst = MakeHolder<TExternalCatSparseValuesHolder>(
                            this->GetId(),
                            SrcData,
                            QuantizedFeaturesInfo
                        );
                    } else {
                        const auto& perfectHash
                            = this->QuantizedFeaturesInfo->GetCategoricalFeaturesPerfectHash(catFeatureIdx);

                        auto getPerfectHashValue = [&] (ui32 srcValue) -> ui32 {
                            auto it = perfectHash.find(srcValue); // find is guaranteed to be thread-safe;
                            Y_ASSERT(it != perfectHash.end());
                            return it->second.Value;
                        };

                        *subsetDst
                            = CreateQuantizedSparseSubset<ui32, EFeatureValuesType::PerfectHashedCategorical>(
                                this->GetId(),
                                this->SrcData,
                                Get<TInvertedIndexedSubset<ui32>>(*subsetInvertedIndexing),
                                getPerfectHashValue,
                                sizeof(ui32) * CHAR_BIT
                            );
                    }
                }
            }
        );
    }

}
