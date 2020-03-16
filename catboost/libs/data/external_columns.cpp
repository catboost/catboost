#include "external_columns.h"

#include <catboost/libs/helpers/double_array_iterator.h>
#include <catboost/private/libs/quantization/utils.h>

#include <util/generic/algorithm.h>
#include <util/generic/utility.h>
#include <util/generic/ylimits.h>
#include <util/system/compiler.h>


namespace NCB {

    THolder<IFeatureValuesHolder> TExternalFloatValuesHolder::CloneWithNewSubsetIndexing(
        const TCloningParams& cloningParams,
        NPar::TLocalExecutor* localExecutor
    ) const {
        Y_UNUSED(localExecutor);
        return MakeHolder<TExternalFloatValuesHolder>(
            GetId(),
            SrcData->CloneWithNewSubsetIndexing(cloningParams.SubsetIndexing),
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
            *SrcData,
            allowNans,
            nanMode,
            GetId(),
            QuantizedFeaturesInfo->GetBorders(floatFeatureIdx),
            MakeArrayRef(result),
            localExecutor
        );

        return NCB::TMaybeOwningArrayHolder<ui8>::CreateOwning(std::move(result));
    }

    IDynamicBlockIteratorBasePtr TExternalFloatValuesHolder::GetBlockIterator(ui32 offset) const {
        const auto floatFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Float>(*this);
        const auto nanMode = QuantizedFeaturesInfo->GetNanMode(floatFeatureIdx);

        // it's ok even if it is learn data, for learn nans are checked at CalcBordersAndNanMode stage
        bool allowNans = (nanMode != ENanMode::Forbidden) ||
            QuantizedFeaturesInfo->GetFloatFeaturesAllowNansInTestOnly();
        auto featureIdx = GetId();

        auto transformer = [floatFeatureIdx, allowNans, nanMode, featureIdx, quantizedFeaturesInfo = QuantizedFeaturesInfo] (TConstArrayRef<float> src, TArrayRef<ui8> dst) {
            QuantizeBlock(
                src,
                allowNans,
                nanMode,
                featureIdx,
                quantizedFeaturesInfo->GetBorders(floatFeatureIdx),
                dst
            );
        };
        return MakeBlockTransformerIterator<ui8>(
            SrcData->GetBlockIterator(offset),
            std::move(transformer)
        );
    }


    THolder<IFeatureValuesHolder> TExternalCatValuesHolder::CloneWithNewSubsetIndexing(
        const TCloningParams& cloningParams,
        NPar::TLocalExecutor* localExecutor
    ) const {
        Y_UNUSED(localExecutor);
        return MakeHolder<TExternalCatValuesHolder>(
            GetId(),
            SrcData->CloneWithNewSubsetIndexing(cloningParams.SubsetIndexing),
            QuantizedFeaturesInfo
        );
    }

    NCB::TMaybeOwningArrayHolder<ui32> TExternalCatValuesHolder::ExtractValues(
        NPar::TLocalExecutor* localExecutor
    ) const {
        TVector<ui32> result;
        result.yresize(GetSize());

        TArrayRef<ui32> resultRef = result;

        const auto catFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Categorical>(
            *this
        );
        const auto& perfectHash = QuantizedFeaturesInfo->GetCategoricalFeaturesPerfectHash(catFeatureIdx);

        SrcData->ParallelForEach(
            [resultRef, &perfectHash] (ui32 idx, ui32 srcValue) {
                resultRef[idx] = perfectHash.Find(srcValue)->Value;
            },
            localExecutor,
            BINARIZATION_BLOCK_SIZE
        );

        return NCB::TMaybeOwningArrayHolder<ui32>::CreateOwning(std::move(result));
    }

    IDynamicBlockIteratorBasePtr TExternalCatValuesHolder::GetBlockIterator(ui32 offset) const {
        const auto catFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Categorical>(
            *this
        );

        auto transformer = [catFeatureIdx, quantizedFeaturesInfo = QuantizedFeaturesInfo] (TConstArrayRef<ui32> src, TArrayRef<ui32> dst) {
            const auto& perfectHash = quantizedFeaturesInfo->GetCategoricalFeaturesPerfectHash(catFeatureIdx);
            for (size_t i : xrange(src.size())) {
                dst[i] = perfectHash.Find(src[i])->Value;
            }
        };
        return MakeBlockTransformerIterator<ui32>(
            SrcData->GetBlockIterator(offset),
            std::move(transformer)
        );
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

    template <class TDstColumn, class TValue, class TSize, class TQuantizeValueFunction>
    static THolder<IFeatureValuesHolder> CreateQuantizedSparseSubset(
        ui32 featureId,
        const TConstPolymorphicValuesSparseArray<TValue, TSize>& srcData,
        const TInvertedIndexedSubset<ui32>& invertedIndexedSubset,
        TQuantizeValueFunction&& quantizeValueFunction,
        ui32 bitsPerKey
    ) {
        TConstArrayRef<ui32> invertedIndicesArray = invertedIndexedSubset.GetMapping();

        TVector<ui32> dstVectorIndexing;
        TVector<typename TDstColumn::TValueType> dstValues;

        srcData.ForEachNonDefault(
            [&](ui32 srcIdx, TValue value) {
                auto dstIdx = invertedIndicesArray[srcIdx];
                if (dstIdx != TInvertedIndexedSubset<ui32>::NOT_PRESENT) {
                    dstVectorIndexing.push_back(dstIdx);
                    dstValues.push_back(quantizeValueFunction(value));
                }
            }
        );

        std::function<TCompressedArray(TVector<typename TDstColumn::TValueType>&&)> createNonDefaultValuesContainer
            = [&] (TVector<typename TDstColumn::TValueType>&& dstValues) {
                return TCompressedArray(
                    dstValues.size(),
                    bitsPerKey,
                    CompressVector<ui64>(dstValues, bitsPerKey)
                );
            };

        return MakeHolder<TSparseCompressedValuesHolderImpl<TDstColumn>>(
            featureId,
            MakeSparseArrayBase<typename TDstColumn::TValueType, TCompressedArray, ui32>(
                invertedIndexedSubset.GetSize(),
                std::move(dstVectorIndexing),
                std::move(dstValues),
                std::move(createNonDefaultValuesContainer),
                /*sparseArrayIndexingType*/ srcData.GetIndexing()->GetType(),
                /*ordered*/ false,
                quantizeValueFunction(srcData.GetDefaultValue())
            )
        );
    }
    ui64 TExternalFloatSparseValuesHolder::EstimateMemoryForCloning(
        const TCloningParams& cloningParams
    ) const {
        const auto floatFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Float>(
            *this
        );
        CB_ENSURE_INTERNAL(cloningParams.InvertedSubsetIndexing.Defined(), "InvertedSubsetIndexing should be defined");
        return EstimateCpuRamLimitForCreateQuantizedSparseSubset<ui8>(
            **cloningParams.InvertedSubsetIndexing,
            SrcData.GetNonDefaultSize(),
            SrcData.GetIndexing()->GetType(),
            CalcHistogramWidthForBorders(QuantizedFeaturesInfo->GetBorders(floatFeatureIdx).size())
        );
    }

    THolder<IFeatureValuesHolder> TExternalFloatSparseValuesHolder::CloneWithNewSubsetIndexing(
        const TCloningParams& cloningParams,
        NPar::TLocalExecutor* localExecutor
    ) const {
        Y_UNUSED(localExecutor);

        CB_ENSURE_INTERNAL(cloningParams.InvertedSubsetIndexing.Defined(), "InvertedSubsetIndexing should be defined");
        const auto& subsetInvertedIndexing = cloningParams.InvertedSubsetIndexing.GetRef();

        const auto floatFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Float>(
            *this
        );

        if (HoldsAlternative<TFullSubset<ui32>>(*subsetInvertedIndexing)) {
            // just clone
            return MakeHolder<TExternalFloatSparseValuesHolder>(
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

            return CreateQuantizedSparseSubset<IQuantizedFloatValuesHolder>(
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


    NCB::TMaybeOwningArrayHolder<ui32> TExternalCatSparseValuesHolder::ExtractValues(
        NPar::TLocalExecutor* localExecutor
    ) const {
        Y_UNUSED(localExecutor);

        const auto catFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Categorical>(
            *this
        );
        const auto& perfectHash = QuantizedFeaturesInfo->GetCategoricalFeaturesPerfectHash(catFeatureIdx);

        const ui32 defaultPerfectHashValue = perfectHash.Find(SrcData.GetDefaultValue())->Value;

        TVector<ui32> result(GetSize(), defaultPerfectHashValue);

        TArrayRef<ui32> resultRef = result;

        SrcData.ForEachNonDefault(
            [=, &perfectHash] (ui32 nonDefaultIdx, ui32 srcValue) {
                resultRef[nonDefaultIdx] = perfectHash.Find(srcValue)->Value;
            }
        );

        return NCB::TMaybeOwningArrayHolder<ui32>::CreateOwning(std::move(result));
    }

    ui64 TExternalCatSparseValuesHolder::EstimateMemoryForCloning(
        const TCloningParams& cloningParams
    ) const {
        return EstimateCpuRamLimitForCreateQuantizedSparseSubset<ui32>(
            **cloningParams.InvertedSubsetIndexing,
            SrcData.GetNonDefaultSize(),
            SrcData.GetIndexing()->GetType(),
            sizeof(ui32) * CHAR_BIT
        );
    }

    THolder<IFeatureValuesHolder> TExternalCatSparseValuesHolder::CloneWithNewSubsetIndexing(
        const TCloningParams& cloningParams,
        NPar::TLocalExecutor* localExecutor
    ) const {
        Y_UNUSED(localExecutor);

        CB_ENSURE_INTERNAL(cloningParams.InvertedSubsetIndexing.Defined(), "InvertedSubsetIndexing should be defined");
        const auto& subsetInvertedIndexing = cloningParams.InvertedSubsetIndexing.GetRef();

        const auto catFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Categorical>(
            *this
        );
        if (HoldsAlternative<TFullSubset<ui32>>(*subsetInvertedIndexing)) {
            // just clone
            return MakeHolder<TExternalCatSparseValuesHolder>(
                this->GetId(),
                SrcData,
                QuantizedFeaturesInfo
            );
        } else {
            const auto& perfectHash
                = this->QuantizedFeaturesInfo->GetCategoricalFeaturesPerfectHash(catFeatureIdx);

            auto getPerfectHashValue = [&] (ui32 srcValue) -> ui32 {
                return perfectHash.Find(srcValue)->Value;
            };

            return CreateQuantizedSparseSubset<IQuantizedCatValuesHolder>(
                this->GetId(),
                this->SrcData,
                Get<TInvertedIndexedSubset<ui32>>(*subsetInvertedIndexing),
                getPerfectHashValue,
                sizeof(ui32) * CHAR_BIT
            );
        }
    }
}
