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
        NPar::ILocalExecutor* localExecutor
    ) const {
        Y_UNUSED(localExecutor);
        return MakeHolder<TExternalFloatValuesHolder>(
            GetId(),
            SrcData->CloneWithNewSubsetIndexing(cloningParams.SubsetIndexing),
            QuantizedFeaturesInfo
        );
    }

    IDynamicBlockIteratorBasePtr TExternalFloatValuesHolder::GetBlockIterator(ui32 offset) const {
        const auto floatFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Float>(*this);
        const auto nanMode = QuantizedFeaturesInfo->GetNanMode(floatFeatureIdx);

        // it's ok even if it is learn data, for learn nans are checked at CalcBordersAndNanMode stage
        bool allowNans = (nanMode != ENanMode::Forbidden) ||
            QuantizedFeaturesInfo->GetFloatFeaturesAllowNansInTestOnly();
        auto featureIdx = GetId();

         auto transformer = [
            allowNans, nanMode, featureIdx,
            bordersArrRef = MakeArrayRef(QuantizedFeaturesInfo->GetBorders(floatFeatureIdx))
        ] (TConstArrayRef<float> src, auto& dst) {
            QuantizeBlock(
                src,
                allowNans,
                nanMode,
                featureIdx,
                bordersArrRef,
                MakeArrayRef(dst)
            );
        };
        if (QuantizedFeaturesInfo->GetBorders(floatFeatureIdx).size() < 256) {
            return MakeBlockTransformerIterator<ui8>(
                SrcData->GetBlockIterator(offset),
                std::move(transformer)
            );
        } else {
            return MakeBlockTransformerIterator<ui16>(
                SrcData->GetBlockIterator(offset),
                std::move(transformer)
            );
        }
    }


    THolder<IFeatureValuesHolder> TExternalCatValuesHolder::CloneWithNewSubsetIndexing(
        const TCloningParams& cloningParams,
        NPar::ILocalExecutor* localExecutor
    ) const {
        Y_UNUSED(localExecutor);
        return MakeHolder<TExternalCatValuesHolder>(
            GetId(),
            SrcData->CloneWithNewSubsetIndexing(cloningParams.SubsetIndexing),
            QuantizedFeaturesInfo
        );
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

    IDynamicBlockIteratorBasePtr TExternalFloatSparseValuesHolder::GetBlockIterator(ui32 offset) const {
        const auto flatFeatureIdx = GetId();
        const auto floatFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Float>(*this);
        const auto nanMode = QuantizedFeaturesInfo->GetNanMode(floatFeatureIdx);
        bool allowNans = (nanMode != ENanMode::Forbidden) ||
            QuantizedFeaturesInfo->GetFloatFeaturesAllowNansInTestOnly();

        TConstArrayRef<float> borders = QuantizedFeaturesInfo->GetBorders(floatFeatureIdx);
        if (borders.size() < 256) {
            auto transformer = [=, quantizedFeaturesInfoHolder = QuantizedFeaturesInfo] (float srcValue) -> ui8 {
                return Quantize<ui8>(flatFeatureIdx, allowNans, nanMode, borders, srcValue);
            };
            return SrcData.GetTransformingBlockIterator<ui8>(std::move(transformer), offset);
        } else {
            auto transformer = [=, quantizedFeaturesInfoHolder = QuantizedFeaturesInfo] (float srcValue) -> ui16 {
                return Quantize<ui16>(flatFeatureIdx, allowNans, nanMode, borders, srcValue);
            };
            return SrcData.GetTransformingBlockIterator<ui16>(std::move(transformer), offset);
        }
    }

    template <class TDst>
    static ui64 EstimateCpuRamLimitForCreateQuantizedSparseSubset(
        const TFeaturesArraySubsetInvertedIndexing& subsetInvertedIndexing,
        ui32 nonDefaultValuesCount,
        ESparseArrayIndexingType sparseArrayIndexingType,
        ui32 dstBitsPerKey
    ) {
        if (std::holds_alternative<TFullSubset<ui32>>(subsetInvertedIndexing)) {
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
                CB_ENSURE(false, "Unexpected sparse array indexing type");
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

    template <class TDstColumn, typename TDstColumnValueType, class TValue, class TSize, class TQuantizeValueFunction>
    static THolder<IFeatureValuesHolder> CreateQuantizedSparseSubset(
        ui32 featureId,
        const TConstPolymorphicValuesSparseArray<TValue, TSize>& srcData,
        const TInvertedIndexedSubset<ui32>& invertedIndexedSubset,
        TQuantizeValueFunction&& quantizeValueFunction,
        ui32 bitsPerKey
    ) {
        TConstArrayRef<ui32> invertedIndicesArray = invertedIndexedSubset.GetMapping();

        TVector<ui32> dstVectorIndexing;
        TVector<TDstColumnValueType> dstValues;

        srcData.ForEachNonDefault(
            [&](ui32 srcIdx, TValue value) {
                auto dstIdx = invertedIndicesArray[srcIdx];
                if (dstIdx != TInvertedIndexedSubset<ui32>::NOT_PRESENT) {
                    dstVectorIndexing.push_back(dstIdx);
                    dstValues.push_back(quantizeValueFunction(value));
                }
            }
        );

        std::function<TCompressedArray(TVector<TDstColumnValueType>&&)> createNonDefaultValuesContainer
            = [&] (TVector<TDstColumnValueType>&& dstValues) {
                return TCompressedArray(
                    dstValues.size(),
                    bitsPerKey,
                    CompressVector<ui64>(dstValues, bitsPerKey)
                );
            };

        return MakeHolder<TSparseCompressedValuesHolderImpl<TDstColumn>>(
            featureId,
            MakeSparseArrayBase<TDstColumnValueType, TCompressedArray, ui32>(
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
        NPar::ILocalExecutor* localExecutor
    ) const {
        Y_UNUSED(localExecutor);

        CB_ENSURE_INTERNAL(cloningParams.InvertedSubsetIndexing.Defined(), "InvertedSubsetIndexing should be defined");
        const auto& subsetInvertedIndexing = cloningParams.InvertedSubsetIndexing.GetRef();

        const auto floatFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Float>(
            *this
        );

        if (std::holds_alternative<TFullSubset<ui32>>(*subsetInvertedIndexing)) {
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

            return CreateQuantizedSparseSubset<IQuantizedFloatValuesHolder, ui8>(
                this->GetId(),
                this->SrcData,
                std::get<TInvertedIndexedSubset<ui32>>(*subsetInvertedIndexing),
                [=] (float srcValue) -> ui8 {
                    return Quantize<ui8>(flatFeatureIdx, allowNans, nanMode, borders, srcValue);
                },

                // TODO(akhropov): fix wide histograms support - MLTOOLS-3758
                sizeof(ui8) * CHAR_BIT
            );
        }
    }

    IDynamicBlockIteratorBasePtr TExternalCatSparseValuesHolder::GetBlockIterator(ui32 offset) const {
        const auto catFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Categorical>(
            *this
        );
        const auto& perfectHash = QuantizedFeaturesInfo->GetCategoricalFeaturesPerfectHash(catFeatureIdx);

        auto transformer = [quantizedFeaturesInfoHolder = QuantizedFeaturesInfo, &perfectHash] (ui32 srcValue) -> ui32 {
            return perfectHash.Find(srcValue)->Value;
        };
        return SrcData.GetTransformingBlockIterator<ui32>(std::move(transformer), offset);
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
        NPar::ILocalExecutor* localExecutor
    ) const {
        Y_UNUSED(localExecutor);

        CB_ENSURE_INTERNAL(cloningParams.InvertedSubsetIndexing.Defined(), "InvertedSubsetIndexing should be defined");
        const auto& subsetInvertedIndexing = cloningParams.InvertedSubsetIndexing.GetRef();

        const auto catFeatureIdx = QuantizedFeaturesInfo->GetPerTypeFeatureIdx<EFeatureType::Categorical>(
            *this
        );
        if (std::holds_alternative<TFullSubset<ui32>>(*subsetInvertedIndexing)) {
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

            return CreateQuantizedSparseSubset<IQuantizedCatValuesHolder, ui32>(
                this->GetId(),
                this->SrcData,
                std::get<TInvertedIndexedSubset<ui32>>(*subsetInvertedIndexing),
                getPerfectHashValue,
                sizeof(ui32) * CHAR_BIT
            );
        }
    }
}
