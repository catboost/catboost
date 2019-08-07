#include "quantization.h"

#include "cat_feature_perfect_hash_helper.h"
#include "columns.h"
#include "external_columns.h"
#include "util.h"

#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/compression.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/resource_constrained_executor.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/text_processing/text_column_builder.h>
#include <catboost/libs/quantization/utils.h>
#include <catboost/libs/quantization_schema/quantize.h>

#include <library/grid_creator/binarization.h>

#include <util/generic/cast.h>
#include <util/generic/maybe.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/random/shuffle.h>
#include <util/system/compiler.h>
#include <util/system/mem_info.h>

#include <limits>
#include <numeric>


namespace NCB {
    static bool NeedToCalcBorders(const TQuantizedFeaturesInfo& quantizedFeaturesInfo) {
        bool needToCalcBorders = false;
        quantizedFeaturesInfo.GetFeaturesLayout()->IterateOverAvailableFeatures<EFeatureType::Float>(
            [&] (TFloatFeatureIdx floatFeatureIdx) {
                if (!quantizedFeaturesInfo.HasBorders(floatFeatureIdx)) {
                    needToCalcBorders = true;
                }
            }
        );

        return needToCalcBorders;
    }


    static TMaybe<TArraySubsetIndexing<ui32>> GetSubsetForBuildBorders(
        const TArraySubsetIndexing<ui32>& srcIndexing,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        EObjectsOrder srcObjectsOrder,
        const TQuantizationOptions& options,
        TRestorableFastRng64* rand
    ) {
        if (NeedToCalcBorders(quantizedFeaturesInfo)) {
            const ui32 objectCount = srcIndexing.Size();
            const ui32 sampleSize = GetSampleSizeForBorderSelectionType(
                objectCount,
                /*TODO(kirillovs): iterate through all per feature binarization settings and select smallest
                 * sample size
                 */
                quantizedFeaturesInfo.GetFloatFeatureBinarization(Max<ui32>()).BorderSelectionType,
                options.MaxSubsetSizeForSlowBuildBordersAlgorithms
            );
            if (sampleSize < objectCount) {
                if (srcObjectsOrder == EObjectsOrder::RandomShuffled) {
                    // just get first sampleSize elements
                    TVector<TSubsetBlock<ui32>> blocks = {TSubsetBlock<ui32>({0, sampleSize}, 0)};
                    return Compose(
                        srcIndexing,
                        TArraySubsetIndexing<ui32>(TRangesSubset<ui32>(sampleSize, std::move(blocks)))
                    );
                } else {
                    TIndexedSubset<ui32> randomShuffle;
                    randomShuffle.yresize(objectCount);
                    std::iota(randomShuffle.begin(), randomShuffle.end(), 0);
                    if (options.CpuCompatibilityShuffleOverFullData) {
                        Shuffle(randomShuffle.begin(), randomShuffle.end(), *rand);
                    } else {
                        for (auto i : xrange(sampleSize)) {
                            std::swap(randomShuffle[i], randomShuffle[rand->Uniform(i, objectCount)]);
                        }
                    }
                    randomShuffle.resize(sampleSize);
                    return Compose(srcIndexing, TArraySubsetIndexing<ui32>(std::move(randomShuffle)));
                }
            }
        }
        return Nothing();
    }


    static ui64 EstimateMemUsageForFloatFeature(
        const TFloatValuesHolder& srcFeature,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TQuantizationOptions& options,
        bool doQuantization, // if false - only calc borders
        bool storeFeaturesDataAsExternalValuesHolder
    ) {
        ui64 result = 0;

        size_t borderCount;

        if (NeedToCalcBorders(quantizedFeaturesInfo)) {
            // sampleSize is computed using defaultBinarizationSettings for now
            const auto& defaultBinarizationSettings
                = quantizedFeaturesInfo.GetFloatFeatureBinarization(Max<ui32>());

            const ui32 sampleSize = GetSampleSizeForBorderSelectionType(
                srcFeature.GetSize(),
                defaultBinarizationSettings.BorderSelectionType,
                options.MaxSubsetSizeForSlowBuildBordersAlgorithms
            );

            result += sizeof(float) * sampleSize; // for copying to srcFeatureValuesForBuildBorders

            const auto& floatFeatureBinarizationSettings
                = quantizedFeaturesInfo.GetFloatFeatureBinarization(srcFeature.GetId());

            borderCount = floatFeatureBinarizationSettings.BorderCount.Get();

            result += NSplitSelection::CalcMemoryForFindBestSplit(
                SafeIntegerCast<int>(borderCount),
                (size_t)sampleSize,
                /*defaultValue*/ Nothing(),
                floatFeatureBinarizationSettings.BorderSelectionType
            );
        } else {
            const TFloatFeatureIdx floatFeatureIdx
                = quantizedFeaturesInfo.GetPerTypeFeatureIdx<EFeatureType::Float>(srcFeature);
            borderCount = quantizedFeaturesInfo.GetBorders(floatFeatureIdx).size();
        }

        if (doQuantization && !storeFeaturesDataAsExternalValuesHolder) {
            // for storing quantized data
            TIndexHelper<ui64> indexHelper(CalcHistogramWidthForBorders(borderCount));
            result += indexHelper.CompressedSize(srcFeature.GetSize()) * sizeof(ui64);
        }

        return result;
    }


    static void CalcBordersAndNanMode(
        const TFloatValuesHolder& srcFeature,
        const TFeaturesArraySubsetIndexing* subsetForBuildBorders,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        ENanMode* nanMode,
        TVector<float>* borders
    ) {
        const auto& binarizationOptions = quantizedFeaturesInfo.GetFloatFeatureBinarization(srcFeature.GetId());

        Y_VERIFY(binarizationOptions.BorderCount > 0);

        // does not contain nans
        TVector<float> srcFeatureValuesForBuildBorders;

        bool hasNans = false;

        if (const auto* denseSrcFeature = dynamic_cast<const TFloatArrayValuesHolder*>(&srcFeature)) {
            TMaybeOwningConstArraySubset<float, ui32> srcFeatureData = denseSrcFeature->GetArrayData();

            TMaybeOwningConstArraySubset<float, ui32> srcDataForBuildBorders(
                srcFeatureData.GetSrc(),
                subsetForBuildBorders
            );

            srcFeatureValuesForBuildBorders.reserve(srcDataForBuildBorders.Size());

            srcDataForBuildBorders.ForEach(
                [&] (ui32 /*idx*/, float value) {
                    if (IsNan(value)) {
                        hasNans = true;
                    } else {
                        srcFeatureValuesForBuildBorders.push_back(value);
                    }
                }
            );
        } else {
            CB_ENSURE_INTERNAL(false, "CalcQuantizationAndNanMode: Unsupported column type");
        }

        CB_ENSURE(
            (binarizationOptions.NanMode != ENanMode::Forbidden) ||
            !hasNans,
            "Feature #" << srcFeature.GetId() << ": There are nan factors and nan values for "
            " float features are not allowed. Set nan_mode != Forbidden."
        );

        int nonNanValuesBorderCount = binarizationOptions.BorderCount;
        if (hasNans) {
            *nanMode = binarizationOptions.NanMode;
            --nonNanValuesBorderCount;
        } else {
            *nanMode = ENanMode::Forbidden;
        }

        THashSet<float> borderSet;

        if (nonNanValuesBorderCount > 0) {
            borderSet = BestSplit(
                srcFeatureValuesForBuildBorders,
                nonNanValuesBorderCount,
                binarizationOptions.BorderSelectionType
            );

            if (borderSet.contains(-0.0f)) { // BestSplit might add negative zeros
                borderSet.erase(-0.0f);
                borderSet.insert(0.0f);
            }
        }

        borders->assign(borderSet.begin(), borderSet.end());
        Sort(borders->begin(), borders->end());

        if (*nanMode == ENanMode::Min) {
            borders->insert(borders->begin(), std::numeric_limits<float>::lowest());
        } else if (*nanMode == ENanMode::Max) {
            borders->push_back(std::numeric_limits<float>::max());
        }
    }


    using TGetBinFunction = std::function<ui32(size_t, size_t)>;

    TGetBinFunction GetQuantizedFloatFeatureFunction(
        const TRawObjectsData& rawObjectsData,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        TFloatFeatureIdx floatFeatureIdx
    ) {
        const auto& denseData = dynamic_cast<const TFloatArrayValuesHolder&>(
            *(rawObjectsData.FloatFeatures[*floatFeatureIdx])
        );
        TConstArrayRef<float> srcRawData = **denseData.GetArrayData().GetSrc();

        auto flatFeatureIdx = quantizedFeaturesInfo.GetFeaturesLayout()->GetExternalFeatureIdx(
            *floatFeatureIdx,
            EFeatureType::Float
        );
        const auto nanMode = quantizedFeaturesInfo.GetNanMode(floatFeatureIdx);
        const bool allowNans = (nanMode != ENanMode::Forbidden) ||
            quantizedFeaturesInfo.GetFloatFeaturesAllowNansInTestOnly();
        TConstArrayRef<float> borders = quantizedFeaturesInfo.GetBorders(floatFeatureIdx);

        return [=](ui32 /*idx*/, ui32 srcIdx) -> ui32 {
            return Quantize<ui32>(flatFeatureIdx, allowNans, nanMode, borders, srcRawData[srcIdx]);
        };
    }

    TGetBinFunction GetQuantizedCatFeatureFunction(
        const TRawObjectsData& rawObjectsData,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        TCatFeatureIdx catFeatureIdx
    ) {
        const auto& denseData = dynamic_cast<const THashedCatArrayValuesHolder&>(
            *(rawObjectsData.CatFeatures[*catFeatureIdx])
        );
        TConstArrayRef<ui32> srcRawData = **denseData.GetArrayData().GetSrc();

        const auto* catFeaturePerfectHashPtr
            = &(quantizedFeaturesInfo.GetCategoricalFeaturesPerfectHash(catFeatureIdx));

        return [srcRawData, catFeaturePerfectHashPtr](ui32 /*idx*/, ui32 srcIdx) -> ui32 {
            return catFeaturePerfectHashPtr->at(srcRawData[srcIdx]).Value;
        };
    }


    TGetNonDefaultValuesMask GetQuantizedFloatNonDefaultValuesMaskFunction(
        const TRawObjectsData& rawObjectsData,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        TFloatFeatureIdx floatFeatureIdx
    ) {
        const auto& denseData = dynamic_cast<const TFloatArrayValuesHolder&>(
            *(rawObjectsData.FloatFeatures[*floatFeatureIdx])
        );
        TConstArrayRef<float> srcRawData = **denseData.GetArrayData().GetSrc();

        auto flatFeatureIdx = quantizedFeaturesInfo.GetFeaturesLayout()->GetExternalFeatureIdx(
            *floatFeatureIdx,
            EFeatureType::Float
        );
        const auto nanMode = quantizedFeaturesInfo.GetNanMode(floatFeatureIdx);
        const bool allowNans = (nanMode != ENanMode::Forbidden) ||
            quantizedFeaturesInfo.GetFloatFeaturesAllowNansInTestOnly();
        float border0 = quantizedFeaturesInfo.GetBorders(floatFeatureIdx)[0];

        return [=](TConstArrayRef<ui32> srcIndices) -> ui64 {
            Y_ASSERT(srcIndices.size() <= (sizeof(ui64) * CHAR_BIT));

            ui64 result = 0;
            for (auto i : xrange(srcIndices.size())) {
                const float srcValue = srcRawData[srcIndices[i]];
                if (IsNan(srcValue)) {
                    CB_ENSURE(
                        allowNans,
                        "There are NaNs in test dataset (feature number "
                        << flatFeatureIdx << ") but there were no NaNs in learn dataset"
                    );
                    if (nanMode == ENanMode::Max) {
                        result |= (ui64(1) << i);
                    }
                } else if (srcValue > border0) {
                    result |= (ui64(1) << i);
                }
            }

            return result;
        };
    }

    TGetNonDefaultValuesMask GetQuantizedCatNonDefaultValuesMaskFunction(
        const TRawObjectsData& rawObjectsData,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        TCatFeatureIdx catFeatureIdx
    ) {
        const auto& denseData = dynamic_cast<const THashedCatArrayValuesHolder&>(
            *(rawObjectsData.CatFeatures[*catFeatureIdx])
        );
        TConstArrayRef<ui32> srcRawData = **denseData.GetArrayData().GetSrc();

        ui32 hashedCatValueMappedTo0 = 0;
        for (const auto& [hashedCatValue, valueAndCount]
             : quantizedFeaturesInfo.GetCategoricalFeaturesPerfectHash(catFeatureIdx))
        {
            if (valueAndCount.Value == 0) {
                hashedCatValueMappedTo0 = hashedCatValue;
                break;
            }
        }

        return [srcRawData, hashedCatValueMappedTo0](TConstArrayRef<ui32> srcIndices) -> ui64 {
            Y_ASSERT(srcIndices.size() <= (sizeof(ui64) * CHAR_BIT));

            ui64 result = 0;
            for (auto i : xrange(srcIndices.size())) {
                if (srcRawData[srcIndices[i]] != hashedCatValueMappedTo0) {
                    result |= (ui64(1) << i);
                }
            }
            return result;
        };
    }


    template <class TBundle>
    THolder<TExclusiveFeatureBundleHolder> BundleFeatures(
        const TExclusiveFeaturesBundle& exclusiveFeaturesBundle,
        const TRawObjectsData& rawObjectsData,
        const TQuantizedForCPUObjectsData& quantizedObjectsData,
        const TFeaturesArraySubsetIndexing& rawDataSubsetIndexing,
        const TFeaturesArraySubsetIndexing* dstSubsetIndexing,
        NPar::TLocalExecutor* localExecutor
    ) {
        const ui32 bitsPerKey = sizeof(TBundle) * CHAR_BIT;
        const ui32 objectCount = rawDataSubsetIndexing.Size();

        TCompressedArray dstStorage
            = TCompressedArray::CreateWithUninitializedData(objectCount, bitsPerKey);

        TBundle* dstData = (TBundle*)dstStorage.GetRawPtr();

        TConstArrayRef<TExclusiveBundlePart> parts = exclusiveFeaturesBundle.Parts;

        const TBundle defaultValue = parts.back().Bounds.End;

        TVector<TGetBinFunction> getBinFunctions;
        for (const auto& part : parts) {
            if (part.FeatureType == EFeatureType::Float) {
                getBinFunctions.push_back(
                    GetQuantizedFloatFeatureFunction(
                        rawObjectsData,
                        *quantizedObjectsData.Data.QuantizedFeaturesInfo,
                        TFloatFeatureIdx(part.FeatureIdx)
                    )
                );
            } else if (part.FeatureType == EFeatureType::Categorical) {
                getBinFunctions.push_back(
                    GetQuantizedCatFeatureFunction(
                        rawObjectsData,
                        *quantizedObjectsData.Data.QuantizedFeaturesInfo,
                        TCatFeatureIdx(part.FeatureIdx)
                    )
                );
            } else {
                CB_ENSURE(false, "Feature bundling is not supported for features of type " << part.FeatureType);
            }
        }

        rawDataSubsetIndexing.ParallelForEach(
            [dstData, parts, defaultValue, &getBinFunctions] (ui32 idx, ui32 srcIdx) {
                for (auto partIdx : xrange(parts.size())) {
                    const ui32 partBin = getBinFunctions[partIdx](idx, srcIdx);
                    if (partBin) {
                        dstData[idx] = (TBundle)(parts[partIdx].Bounds.Begin + partBin - 1);
                        return;
                    }
                }
                dstData[idx] = defaultValue;
            },
            localExecutor
        );

        return MakeHolder<TExclusiveFeatureBundleArrayHolder>(
            0, // unused
            std::move(dstStorage),
            dstSubsetIndexing
        );
    }

    static void ScheduleBundleFeatures(
        const TFeaturesArraySubsetIndexing& rawDataSubsetIndexing,
        bool clearSrcObjectsData,
        const TFeaturesArraySubsetIndexing* quantizedDataSubsetIndexing,
        NPar::TLocalExecutor* localExecutor,
        TResourceConstrainedExecutor* resourceConstrainedExecutor,
        TRawObjectsData* rawObjectsData,
        TQuantizedForCPUObjectsData* quantizedObjectsData
    ) {
        const auto& metaData = quantizedObjectsData->ExclusiveFeatureBundlesData.MetaData;

        const auto bundleCount = metaData.size();
        quantizedObjectsData->ExclusiveFeatureBundlesData.SrcData.resize(bundleCount);

        const ui32 objectCount = rawDataSubsetIndexing.Size();

        for (auto bundleIdx : xrange(bundleCount)) {
            resourceConstrainedExecutor->Add(
                {
                    objectCount * metaData[bundleIdx].SizeInBytes,

                    [rawDataSubsetIndexingPtr = &rawDataSubsetIndexing,
                     clearSrcObjectsData,
                     quantizedDataSubsetIndexing,
                     localExecutor,
                     rawObjectsData,
                     quantizedObjectsData,
                     bundleIdx] () {

                        auto& exclusiveFeatureBundlesData = quantizedObjectsData->ExclusiveFeatureBundlesData;
                        const auto& bundleMetaData = exclusiveFeatureBundlesData.MetaData[bundleIdx];
                        auto& bundleData = exclusiveFeatureBundlesData.SrcData[bundleIdx];

                        switch (bundleMetaData.SizeInBytes) {
                            case 1:
                                bundleData = BundleFeatures<ui8>(
                                    bundleMetaData,
                                    *rawObjectsData,
                                    *quantizedObjectsData,
                                    *rawDataSubsetIndexingPtr,
                                    quantizedDataSubsetIndexing,
                                    localExecutor
                                );
                                break;
                            case 2:
                                bundleData = BundleFeatures<ui16>(
                                    bundleMetaData,
                                    *rawObjectsData,
                                    *quantizedObjectsData,
                                    *rawDataSubsetIndexingPtr,
                                    quantizedDataSubsetIndexing,
                                    localExecutor
                                );
                                break;
                            default:
                                CB_ENSURE_INTERNAL(
                                    false,
                                    "unsupported Bundle SizeInBytes = " << bundleMetaData.SizeInBytes
                                );
                        }

                        for (auto partIdx : xrange(bundleMetaData.Parts.size())) {
                            const auto& part = bundleMetaData.Parts[partIdx];
                            if (part.FeatureType == EFeatureType::Float) {
                                quantizedObjectsData->Data.FloatFeatures[part.FeatureIdx].Reset(
                                    new TQuantizedFloatBundlePartValuesHolder(
                                        rawObjectsData->FloatFeatures[part.FeatureIdx]->GetId(),
                                        bundleData.Get(),
                                        part.Bounds
                                    )
                                );
                                if (clearSrcObjectsData) {
                                    rawObjectsData->FloatFeatures[part.FeatureIdx].Destroy();
                                }
                            } else if (part.FeatureType == EFeatureType::Categorical) {
                                quantizedObjectsData->Data.CatFeatures[part.FeatureIdx].Reset(
                                    new TQuantizedCatBundlePartValuesHolder(
                                        rawObjectsData->CatFeatures[part.FeatureIdx]->GetId(),
                                        bundleData.Get(),
                                        part.Bounds
                                    )
                                );
                                if (clearSrcObjectsData) {
                                    rawObjectsData->CatFeatures[part.FeatureIdx].Destroy();
                                }
                            } else {
                                CB_ENSURE(false, "Feature bundling is not supported for features of type " << part.FeatureType);
                            }
                        }
                    }
                }
            );
        }
    }

    template <
        class IQuantizedValuesHolder,
        class TExternalValuesHolder,
        class TSrc,
        EFeatureValuesType SrcFeatureValuesType>
    static THolder<IQuantizedValuesHolder> MakeExternalValuesHolder(
        const TTypedFeatureValuesHolder<TSrc, SrcFeatureValuesType>& srcFeature,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo
    ) {
        using TDenseSrcFeature = TArrayValuesHolder<TSrc, SrcFeatureValuesType>;

        if (const auto* denseSrcFeature = dynamic_cast<const TDenseSrcFeature*>(&srcFeature)){
            const auto arrayData = denseSrcFeature->GetArrayData();

            return MakeHolder<TExternalValuesHolder>(
                denseSrcFeature->GetId(),
                *arrayData.GetSrc(),
                arrayData.GetSubsetIndexing(),
                quantizedFeaturesInfo
            );
        } else {
            CB_ENSURE_INTERNAL(false, "MakeExternalValuesHolder: unsupported src feature type");
        }
    }

    static THolder<IQuantizedFloatValuesHolder> MakeQuantizedFloatColumn(
        const TFloatValuesHolder& srcFeature,
        ENanMode nanMode,
        bool allowNans,
        TConstArrayRef<float> borders,
        const TFeaturesArraySubsetIndexing* dstSubsetIndexing,
        NPar::TLocalExecutor* localExecutor
    ) {
        const ui32 bitsPerKey = CalcHistogramWidthForBorders(borders.size());

        auto quantizeNonDefaultValues
            = [&] (TMaybeOwningConstArraySubset<float, ui32> nonDefaultValues) -> TCompressedArray {

                TCompressedArray quantizedDataStorage
                    = TCompressedArray::CreateWithUninitializedData(nonDefaultValues.Size(), bitsPerKey);

                auto quantizeNonDefaultValuesInner = [&] (auto* dstBegin) {
                    Quantize(
                        nonDefaultValues,
                        allowNans,
                        nanMode,
                        srcFeature.GetId(),
                        borders,
                        MakeArrayRef(dstBegin, nonDefaultValues.Size()),
                        localExecutor
                    );
                };

                switch (bitsPerKey) {
                    case 8:
                        quantizeNonDefaultValuesInner((ui8*)quantizedDataStorage.GetRawPtr());
                        break;
                    case 16:
                        quantizeNonDefaultValuesInner((ui16*)quantizedDataStorage.GetRawPtr());
                        break;
                    default:
                        CB_ENSURE_INTERNAL(
                            false,
                            "MakeQuantizedFloatColumn: unsupported " << LabeledOutput(bitsPerKey)
                        );
                }

                return quantizedDataStorage;
            };

        if (const auto* denseSrcFeature = dynamic_cast<const TFloatArrayValuesHolder*>(&srcFeature)){
            return MakeHolder<TQuantizedFloatValuesHolder>(
                srcFeature.GetId(),
                quantizeNonDefaultValues(denseSrcFeature->GetArrayData()),
                dstSubsetIndexing
            );
        } else {
            CB_ENSURE_INTERNAL(false, "MakeQuantizedFloatColumn: unsupported src feature type");
        }
        Y_UNREACHABLE();
    }


    static THolder<IQuantizedCatValuesHolder> MakeQuantizedCatColumn(
        const THashedCatValuesHolder& srcFeature,
        const TCatFeaturePerfectHash& perfectHash,
        const TFeaturesArraySubsetIndexing* dstSubsetIndexing,
        NPar::TLocalExecutor* localExecutor
    ) {
        // TODO(akhropov): support other bitsPerKey. MLTOOLS-2425
        const ui32 bitsPerKey = 32;

        auto quantizeValues
            = [&] (TMaybeOwningConstArraySubset<ui32, ui32> srcFeatureData) -> TCompressedArray {
                TCompressedArray quantizedDataStorage
                    = TCompressedArray::CreateWithUninitializedData(srcFeatureData.Size(), bitsPerKey);

                TArrayRef<ui32> quantizedData = quantizedDataStorage.GetRawArray<ui32>();

                srcFeatureData.ParallelForEach(
                    [&] (ui32 idx, ui32 srcValue) {
                        auto it = perfectHash.find(srcValue); // find is guaranteed to be thread-safe

                        // TODO(akhropov): replace by assert for performance?
                        CB_ENSURE(
                            it != perfectHash.end(),
                            "Error: hash for feature #" << srcFeature.GetId() << " was not found " << srcValue
                        );

                        quantizedData[idx] = it->second.Value;
                    },
                    localExecutor,
                    BINARIZATION_BLOCK_SIZE
                );

                return quantizedDataStorage;
            };

        if (const auto* arraySrcFeature = dynamic_cast<const THashedCatArrayValuesHolder*>(&srcFeature)) {
            return MakeHolder<TQuantizedCatValuesHolder>(
                srcFeature.GetId(),
                quantizeValues(arraySrcFeature->GetArrayData()),
                dstSubsetIndexing
            );
        } else {
            CB_ENSURE_INTERNAL(false, "MakeQuantizedCatColumn: unsupported src feature type");
        }
        Y_UNREACHABLE();
    }


    static void ScheduleNonBundledAndNonBinaryFeatures(
        const TFeaturesArraySubsetIndexing& rawDataSubsetIndexing,
        bool clearSrcObjectsData,
        const TFeaturesArraySubsetIndexing* quantizedDataSubsetIndexing,
        NPar::TLocalExecutor* localExecutor,
        TResourceConstrainedExecutor* resourceConstrainedExecutor,
        TRawObjectsData* rawObjectsData,
        TQuantizedForCPUObjectsData* quantizedObjectsData
    ) {
        const ui32 objectCount = rawDataSubsetIndexing.Size();

        const auto& featuresLayout = *quantizedObjectsData->Data.QuantizedFeaturesInfo->GetFeaturesLayout();

        auto isBinaryPackedOrBundled = [&] (EFeatureType featureType, ui32 perTypeFeatureIdx) {
            const ui32 flatFeatureIdx = featuresLayout.GetExternalFeatureIdx(perTypeFeatureIdx, featureType);

            if (quantizedObjectsData
                    ->ExclusiveFeatureBundlesData.FlatFeatureIndexToBundlePart[flatFeatureIdx])
            {
                return true;
            }
            if (quantizedObjectsData
                    ->PackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex[flatFeatureIdx])
            {
                return true;
            }
            return false;
        };

        featuresLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
            [&] (TFloatFeatureIdx floatFeatureIdx) {
                if (isBinaryPackedOrBundled(EFeatureType::Float, *floatFeatureIdx)) {
                    return;
                }

                resourceConstrainedExecutor->Add(
                    {
                        objectCount * sizeof(ui8),

                        [clearSrcObjectsData,
                         quantizedDataSubsetIndexing,
                         localExecutor,
                         rawObjectsData,
                         quantizedObjectsData,
                         floatFeatureIdx] () {
                            const auto& quantizedFeaturesInfo
                                = *quantizedObjectsData->Data.QuantizedFeaturesInfo;

                            const auto nanMode = quantizedFeaturesInfo.GetNanMode(floatFeatureIdx);
                            const bool allowNans = (nanMode != ENanMode::Forbidden) ||
                                quantizedFeaturesInfo.GetFloatFeaturesAllowNansInTestOnly();

                            quantizedObjectsData->Data.FloatFeatures[*floatFeatureIdx]
                                = MakeQuantizedFloatColumn(
                                    *(rawObjectsData->FloatFeatures[*floatFeatureIdx]),
                                    nanMode,
                                    allowNans,
                                    quantizedFeaturesInfo.GetBorders(floatFeatureIdx),
                                    quantizedDataSubsetIndexing,
                                    localExecutor
                                );

                            if (clearSrcObjectsData) {
                                rawObjectsData->FloatFeatures[*floatFeatureIdx].Destroy();
                            }
                        }
                    }
                );
            }
        );

        featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
            [&] (TCatFeatureIdx catFeatureIdx) {
                if (isBinaryPackedOrBundled(EFeatureType::Categorical, *catFeatureIdx)) {
                    return;
                }

                resourceConstrainedExecutor->Add(
                    {
                        objectCount * sizeof(ui32),

                        [clearSrcObjectsData,
                         quantizedDataSubsetIndexing,
                         localExecutor,
                         rawObjectsData,
                         quantizedObjectsData,
                         catFeatureIdx] () {
                            const auto& quantizedFeaturesInfo
                                = *quantizedObjectsData->Data.QuantizedFeaturesInfo;

                            quantizedObjectsData->Data.CatFeatures[*catFeatureIdx]
                                = MakeQuantizedCatColumn(
                                    *(rawObjectsData->CatFeatures[*catFeatureIdx]),
                                    quantizedFeaturesInfo.GetCategoricalFeaturesPerfectHash(catFeatureIdx),
                                    quantizedDataSubsetIndexing,
                                    localExecutor
                                );

                            if (clearSrcObjectsData) {
                                rawObjectsData->CatFeatures[*catFeatureIdx].Destroy();
                            }
                        }
                    }
                );
            }
        );
    }


    /* arguments are idx, srcIdx from rawDataSubsetIndexing
     * each function returns TBinaryFeaturesPack = 0 or 1
     */
    using TGetBitFunction = std::function<TBinaryFeaturesPack(size_t, size_t)>;

    static TGetBitFunction GetBinaryFloatFeatureFunction(
        const TRawObjectsData& rawObjectsData,
        const TQuantizedObjectsData& quantizedObjectsData,
        TFloatFeatureIdx floatFeatureIdx
    ) {
        const auto& denseData = dynamic_cast<const TFloatArrayValuesHolder&>(
            *(rawObjectsData.FloatFeatures[*floatFeatureIdx])
        );
        TConstArrayRef<float> srcRawData = **denseData.GetArrayData().GetSrc();

        float border = quantizedObjectsData.QuantizedFeaturesInfo->GetBorders(floatFeatureIdx)[0];

        return [srcRawData, border](ui32 /*idx*/, ui32 srcIdx) -> TBinaryFeaturesPack {
            return srcRawData[srcIdx] >= border ?
                TBinaryFeaturesPack(1) : TBinaryFeaturesPack(0);
        };
    }

    static TGetBitFunction GetBinaryCatFeatureFunction(
        const TRawObjectsData& rawObjectsData,
        const TQuantizedObjectsData& quantizedObjectsData,
        TCatFeatureIdx catFeatureIdx
    ) {
        const auto& denseData = dynamic_cast<const THashedCatArrayValuesHolder&>(
            *(rawObjectsData.CatFeatures[*catFeatureIdx])
        );
        TConstArrayRef<ui32> srcRawData = **denseData.GetArrayData().GetSrc();

        const auto& catFeaturePerfectHash
            = quantizedObjectsData.QuantizedFeaturesInfo->GetCategoricalFeaturesPerfectHash(catFeatureIdx);
        Y_ASSERT(catFeaturePerfectHash.size() == 2);

        ui32 hashedCatValueFor1;
        for (const auto& [hashedCatValue, remappedValueAndCount] : catFeaturePerfectHash) {
            if (remappedValueAndCount.Value == ui32(1)) {
                hashedCatValueFor1 = hashedCatValue;
            }
        }

        return [srcRawData, hashedCatValueFor1](ui32 /*idx*/, ui32 srcIdx) -> TBinaryFeaturesPack {
            return srcRawData[srcIdx] == hashedCatValueFor1 ?
                TBinaryFeaturesPack(1) : TBinaryFeaturesPack(0);
        };
    }

    template <class T, EFeatureValuesType FeatureValuesType>
    static void SetBinaryFeatureColumn(
        ui32 featureId,
        const TBinaryPacksHolder* packsData,
        ui8 bitIdx,
        THolder<TTypedFeatureValuesHolder<T, FeatureValuesType>>* featureColumn
    ) {
        featureColumn->Reset(
            new TPackedBinaryValuesHolderImpl<T, FeatureValuesType>(featureId, packsData, bitIdx)
        );
    }

    static void ScheduleBinarizeFeatures(
        const TFeaturesArraySubsetIndexing& rawDataSubsetIndexing,
        bool clearSrcObjectsData,
        const TFeaturesArraySubsetIndexing* quantizedDataSubsetIndexing,
        NPar::TLocalExecutor* localExecutor,
        TResourceConstrainedExecutor* resourceConstrainedExecutor,
        TRawObjectsData* rawObjectsData,
        TQuantizedForCPUObjectsData* quantizedObjectsData
    ) {
        const ui32 objectCount = rawDataSubsetIndexing.Size();

        for (auto packIdx : xrange(quantizedObjectsData->PackedBinaryFeaturesData.SrcData.size())) {
            resourceConstrainedExecutor->Add(
                {
                    objectCount * sizeof(TBinaryFeaturesPack),

                    [rawDataSubsetIndexingPtr = &rawDataSubsetIndexing,
                     clearSrcObjectsData,
                     quantizedDataSubsetIndexing,
                     localExecutor,
                     rawObjectsData,
                     quantizedObjectsData,
                     objectCount,
                     packIdx] () {
                        const size_t bitsPerPack = sizeof(TBinaryFeaturesPack) * CHAR_BIT;

                        auto& packedBinaryFeaturesData = quantizedObjectsData->PackedBinaryFeaturesData;

                        const auto& packedBinaryToSrcIndex = packedBinaryFeaturesData.PackedBinaryToSrcIndex;

                        TCompressedArray dstPackedFeaturesDataStorage
                            = TCompressedArray::CreateWithUninitializedData(objectCount, bitsPerPack);

                        TArrayRef<TBinaryFeaturesPack> dstPackedFeaturesData
                            = dstPackedFeaturesDataStorage.GetRawArray<TBinaryFeaturesPack>();

                        TVector<TGetBitFunction> getBitFunctions;

                        size_t startIdx = size_t(packIdx)*bitsPerPack;
                        size_t endIdx = Min(startIdx + bitsPerPack, packedBinaryToSrcIndex.size());

                        auto endIt = packedBinaryToSrcIndex.begin() + endIdx;
                        for (auto it = packedBinaryToSrcIndex.begin() + startIdx; it != endIt; ++it) {
                            if (it->first == EFeatureType::Float) {
                                getBitFunctions.push_back(
                                    GetBinaryFloatFeatureFunction(
                                        *rawObjectsData,
                                        quantizedObjectsData->Data,
                                        TFloatFeatureIdx(it->second)
                                    )
                                );
                            } else if (it->first == EFeatureType::Categorical) {
                                getBitFunctions.push_back(
                                    GetBinaryCatFeatureFunction(
                                        *rawObjectsData,
                                        quantizedObjectsData->Data,
                                        TCatFeatureIdx(it->second)
                                    )
                                );
                            }
                        }

                        rawDataSubsetIndexingPtr->ParallelForEach(
                            [&] (ui32 idx, ui32 srcIdx) {
                                TBinaryFeaturesPack pack = 0;
                                for (auto bitIdx : xrange(getBitFunctions.size())) {
                                    pack |= (getBitFunctions[bitIdx](idx, srcIdx) << bitIdx);
                                }
                                dstPackedFeaturesData[idx] = pack;
                            },
                            localExecutor
                        );

                        packedBinaryFeaturesData.SrcData[packIdx] = MakeHolder<TBinaryPacksArrayHolder>(
                            0, // unused
                            std::move(dstPackedFeaturesDataStorage),
                            quantizedDataSubsetIndexing
                        );

                        for (ui8 bitIdx = 0; bitIdx < (endIdx - startIdx); ++bitIdx) {
                            auto it = packedBinaryToSrcIndex.begin() + startIdx + bitIdx;

                            if (it->first == EFeatureType::Float) {
                                SetBinaryFeatureColumn(
                                    rawObjectsData->FloatFeatures[it->second]->GetId(),
                                    packedBinaryFeaturesData.SrcData[packIdx].Get(),
                                    bitIdx,
                                    &(quantizedObjectsData->Data.FloatFeatures[it->second])
                                );
                                if (clearSrcObjectsData) {
                                    rawObjectsData->FloatFeatures[it->second].Destroy();
                                }
                            } else if (it->first == EFeatureType::Categorical) {
                                SetBinaryFeatureColumn(
                                    rawObjectsData->CatFeatures[it->second]->GetId(),
                                    packedBinaryFeaturesData.SrcData[packIdx].Get(),
                                    bitIdx,
                                    &(quantizedObjectsData->Data.CatFeatures[it->second])
                                );
                                if (clearSrcObjectsData) {
                                    rawObjectsData->CatFeatures[it->second].Destroy();
                                }
                            }
                        }
                    }
                }
            );
        }
    }


    static void ProcessFloatFeature(
        TFloatFeatureIdx floatFeatureIdx,
        const TFloatValuesHolder& srcFeature,
        const TFeaturesArraySubsetIndexing* subsetForBuildBorders,
        const TQuantizationOptions& options,
        bool calcBordersAndNanModeOnly,
        bool storeFeaturesDataAsExternalValuesHolder,
        const TFeaturesArraySubsetIndexing* dstSubsetIndexing,  // can be nullptr if generateBordersOnly
        NPar::TLocalExecutor* localExecutor,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        THolder<IQuantizedFloatValuesHolder>* dstQuantizedFeature // can be nullptr if generateBordersOnly
    ) {
        bool calculateNanMode = true;
        ENanMode nanMode = ENanMode::Forbidden;

        bool calculateBorders = true;
        TConstArrayRef<float> borders;
        TVector<float> calculatedBorders;

        {
            TReadGuard readGuard(quantizedFeaturesInfo->GetRWMutex());
            if (quantizedFeaturesInfo->HasNanMode(floatFeatureIdx)) {
                calculateNanMode = false;
                nanMode = quantizedFeaturesInfo->GetNanMode(floatFeatureIdx);
            }
            if (quantizedFeaturesInfo->HasBorders(floatFeatureIdx)) {
                calculateBorders = false;
                borders = quantizedFeaturesInfo->GetBorders(floatFeatureIdx);
            }
        }

        CB_ENSURE_INTERNAL(
            calculateNanMode == calculateBorders,
            "Feature #" << srcFeature.GetId()
            << ": NanMode and borders must be specified or not specified together"
        );

        if (calculateNanMode || calculateBorders) {
            CalcBordersAndNanMode(
                srcFeature,
                subsetForBuildBorders,
                *quantizedFeaturesInfo,
                &nanMode,
                &calculatedBorders
            );

            borders = calculatedBorders;
        }

        if (!calcBordersAndNanModeOnly && !borders.empty()) {
            if (storeFeaturesDataAsExternalValuesHolder) {
                // use GPU-only external columns
                *dstQuantizedFeature =
                    MakeExternalValuesHolder<
                        IQuantizedFloatValuesHolder,
                        TExternalFloatValuesHolder>(
                            srcFeature,
                            quantizedFeaturesInfo
                        );
            } else if (!options.CpuCompatibleFormat ||
                !options.PackBinaryFeaturesForCpu ||
                (borders.size() > 1)) // binary features are binarized later by packs
            {
                // it's ok even if it is learn data, for learn nans are checked at CalcBordersAndNanMode stage
                bool allowNans = (nanMode != ENanMode::Forbidden) ||
                    quantizedFeaturesInfo->GetFloatFeaturesAllowNansInTestOnly();

                *dstQuantizedFeature = MakeQuantizedFloatColumn(
                    srcFeature,
                    nanMode,
                    allowNans,
                    borders,
                    dstSubsetIndexing,
                    localExecutor
                );
            }
        }

        if (calculateNanMode || calculateBorders) {
            TWriteGuard writeGuard(quantizedFeaturesInfo->GetRWMutex());

            if (calculateNanMode) {
                quantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanMode);
            }
            if (calculateBorders) {
                if (calculatedBorders.empty()) {
                    CATBOOST_DEBUG_LOG << "Float Feature #" << srcFeature.GetId() << " is empty" << Endl;

                    quantizedFeaturesInfo->GetFeaturesLayout()->IgnoreExternalFeature(srcFeature.GetId());
                }

                quantizedFeaturesInfo->SetBorders(floatFeatureIdx, std::move(calculatedBorders));
            }
        }
    }


    static ui64 EstimateMaxMemUsageForCatFeature(
        ui32 objectCount,
        bool storeFeaturesDataAsExternalValuesHolder
    ) {
        ui64 result = 0;

        constexpr ui32 ESTIMATED_FEATURES_PERFECT_HASH_MAP_NODE_SIZE = 32;

        // assuming worst-case that all values will be added to Features Perfect Hash as new.
        result += ESTIMATED_FEATURES_PERFECT_HASH_MAP_NODE_SIZE * objectCount;

        if (!storeFeaturesDataAsExternalValuesHolder) {
            // for storing quantized data
            // TODO(akhropov): support other bitsPerKey. MLTOOLS-2425
            result += sizeof(ui32) * objectCount;
        }

        return result;
    }


    static void ProcessCatFeature(
        TCatFeatureIdx catFeatureIdx,
        const THashedCatValuesHolder& srcFeature,
        bool updatePerfectHashOnly,
        bool storeFeaturesDataAsExternalValuesHolder,
        bool mapMostFrequentValueTo0,
        const TFeaturesArraySubsetIndexing* dstSubsetIndexing,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        THolder<IQuantizedCatValuesHolder>* dstQuantizedFeature
    ) {

        // GPU-only external columns
        const bool quantizeData = !updatePerfectHashOnly && !storeFeaturesDataAsExternalValuesHolder;


        TCompressedArray quantizedDataStorage;

        auto processValues = [&] (TMaybeOwningConstArraySubset<ui32, ui32> srcFeatureData) {
            TArrayRef<ui32> quantizedDataValue;

            if (quantizeData) {
                // TODO(akhropov): support other bitsPerKey. MLTOOLS-2425
                const ui32 bitsPerKey = 32;

                quantizedDataStorage
                    = TCompressedArray::CreateWithUninitializedData(srcFeatureData.Size(), bitsPerKey);
                quantizedDataValue = quantizedDataStorage.GetRawArray<ui32>();
            }

            TCatFeaturesPerfectHashHelper catFeaturesPerfectHashHelper(quantizedFeaturesInfo);

            catFeaturesPerfectHashHelper.UpdatePerfectHashAndMaybeQuantize(
                catFeatureIdx,
                srcFeatureData,
                mapMostFrequentValueTo0,
                quantizeData ? TMaybe<TArrayRef<ui32>*>(&quantizedDataValue) : Nothing()
            );
        };

        if (const auto* arraySrcFeature = dynamic_cast<const THashedCatArrayValuesHolder*>(&srcFeature)) {
            processValues(arraySrcFeature->GetArrayData());
        } else {
            CB_ENSURE_INTERNAL(false, "ProcessCatFeature: unsupported src feature type");
        }


        auto uniqueValuesCounts = quantizedFeaturesInfo->GetUniqueValuesCounts(catFeatureIdx);
        if (uniqueValuesCounts.OnLearnOnly > 1) {
            if (!updatePerfectHashOnly) {
                if (storeFeaturesDataAsExternalValuesHolder) {
                    *dstQuantizedFeature =
                        MakeExternalValuesHolder<
                            IQuantizedCatValuesHolder,
                            TExternalCatValuesHolder>(
                                srcFeature,
                                quantizedFeaturesInfo
                            );
                } else if (quantizedDataStorage.GetSize()) {
                    // was initialized at first pass
                    *dstQuantizedFeature = MakeHolder<TQuantizedCatValuesHolder>(
                        srcFeature.GetId(),
                        std::move(quantizedDataStorage),
                        dstSubsetIndexing
                    );
                }
            }
        } else {
            CATBOOST_DEBUG_LOG << "Categorical Feature #" << srcFeature.GetId() << " is constant" << Endl;

            quantizedFeaturesInfo->GetFeaturesLayout()->IgnoreExternalFeature(srcFeature.GetId());
        }
    }


    static void ProcessTextFeature(
        TTextFeatureIdx textFeatureIdx,
        const TStringTextValuesHolder& srcFeature,
        const TFeaturesArraySubsetIndexing* dstSubsetIndexing,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        THolder<TTokenizedTextValuesHolder>* dstQuantizedFeature
    ) {
        const TStringTextArrayValuesHolder& srcDenseFeature
            = dynamic_cast<const TStringTextArrayValuesHolder&>(srcFeature);

        TMaybeOwningConstArraySubset<TString, ui32> srcFeatureData = srcDenseFeature.GetArrayData();
        const auto &textProcessingOptions = quantizedFeaturesInfo->GetTextFeatureProcessing(srcFeature.GetId());
        const TTokenizerPtr tokenizer = CreateTokenizer(textProcessingOptions.TokenizerType);

        if (!quantizedFeaturesInfo->HasDictionary(textFeatureIdx)) {
            TDictionaryPtr dictionary = CreateDictionary(TIterableTextFeature(srcFeatureData), textProcessingOptions, tokenizer);
            quantizedFeaturesInfo->SetDictionary(textFeatureIdx, dictionary);
        }

        const TDictionaryPtr dictionary = quantizedFeaturesInfo->GetDictionary(textFeatureIdx);
        TTextColumnBuilder textColumnBuilder(tokenizer, dictionary, srcFeatureData.Size());
        srcFeatureData.ForEach([&](ui32 index, TStringBuf phrase) {
            textColumnBuilder.AddText(index, phrase);
        });

        *dstQuantizedFeature = MakeHolder<TTokenizedTextArrayValuesHolder>(
            srcFeature.GetId(),
            TTextColumn::CreateOwning(textColumnBuilder.Build()),
            dstSubsetIndexing
        );
    }


    static bool IsFloatFeatureToBeBinarized(
        const TQuantizationOptions& options,
        TQuantizedFeaturesInfo& quantizedFeaturesInfo, // non const because of GetRWMutex
        TFloatFeatureIdx floatFeatureIdx
    ) {
        if (!options.CpuCompatibleFormat || !options.PackBinaryFeaturesForCpu) {
            return false;
        }

        {
            TReadGuard guard(quantizedFeaturesInfo.GetRWMutex());

            if (quantizedFeaturesInfo.GetFeaturesLayout()->GetInternalFeatureMetaInfo(
                    *floatFeatureIdx,
                    EFeatureType::Float
                ).IsAvailable &&
                (quantizedFeaturesInfo.GetBorders(floatFeatureIdx).size() == 1))
            {
                return true;
            }
        }
        return false;
    }

    static bool IsCatFeatureToBeBinarized(
        const TQuantizationOptions& options,
        TQuantizedFeaturesInfo& quantizedFeaturesInfo, // non const because of GetRWMutex
        TCatFeatureIdx catFeatureIdx
    ) {
        if (!options.CpuCompatibleFormat || !options.PackBinaryFeaturesForCpu) {
            return false;
        }

        {
            TReadGuard guard(quantizedFeaturesInfo.GetRWMutex());

            if (quantizedFeaturesInfo.GetFeaturesLayout()->GetInternalFeatureMetaInfo(
                    *catFeatureIdx,
                    EFeatureType::Categorical
                ).IsAvailable &&
                (quantizedFeaturesInfo.GetUniqueValuesCounts(catFeatureIdx).OnAll == 2))
            {
                return true;
            }
        }
        return false;
    }


    // this is a helper class needed for friend declarations
    class TQuantizationImpl {
    public:
        // returns nullptr if generateBordersOnly
        static TQuantizedDataProviderPtr Do(
            const TQuantizationOptions& options,
            TRawDataProviderPtr rawDataProvider,
            TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
            bool calcBordersAndNanModeOnly,
            TRestorableFastRng64* rand,
            NPar::TLocalExecutor* localExecutor
        ) {
            CB_ENSURE_INTERNAL(
                options.CpuCompatibleFormat || options.GpuCompatibleFormat,
                "TQuantizationOptions: at least one of CpuCompatibleFormat or GpuCompatibleFormat"
                "options must be true"
            );

            auto& srcObjectsCommonData = rawDataProvider->ObjectsData->CommonData;

            auto featuresLayout = quantizedFeaturesInfo->GetFeaturesLayout();

            CheckCompatibleForApply(
                *featuresLayout,
                *(srcObjectsCommonData.FeaturesLayout),
                "data to quantize"
            );

            const bool clearSrcData = rawDataProvider->RefCount() <= 1;
            const bool clearSrcObjectsData = clearSrcData &&
                (rawDataProvider->ObjectsData->RefCount() <= 1);

            const bool bundleExclusiveFeatures =
                options.CpuCompatibleFormat && options.BundleExclusiveFeaturesForCpu;

            /*
             * If these conditions are satisfied quantized features data is only needed for GPU
             *  so it is possible not to store all quantized features bins in CPU RAM
             *  but generate these quantized feature bin values from raw feature values on the fly
             *  just before copying data to GPU memory.
             *  Returned TQuantizedObjectsDataProvider will contain
             *  TExternalFloatValuesHolders and TExternalCatValuesHolders in features data holders.
             */
            const bool storeFeaturesDataAsExternalValuesHolders = !options.CpuCompatibleFormat &&
                !clearSrcObjectsData &&
                !featuresLayout->GetTextFeatureCount();

            TObjectsGroupingPtr objectsGrouping = rawDataProvider->ObjectsGrouping;

            // already composed with rawDataProvider's Subset
            TMaybe<TArraySubsetIndexing<ui32>> subsetForBuildBorders = GetSubsetForBuildBorders(
                *(srcObjectsCommonData.SubsetIndexing),
                *quantizedFeaturesInfo,
                srcObjectsCommonData.Order,
                options,
                rand
            );

            TMaybe<TQuantizedForCPUBuilderData> data;
            TAtomicSharedPtr<TArraySubsetIndexing<ui32>> subsetIndexing;

            if (!calcBordersAndNanModeOnly) {
                data.ConstructInPlace();

                auto flatFeatureCount = featuresLayout->GetExternalFeatureCount();
                data->ObjectsData.PackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex.resize(
                    flatFeatureCount
                );
                data->ObjectsData.ExclusiveFeatureBundlesData.FlatFeatureIndexToBundlePart.resize(
                    flatFeatureCount
                );

                data->ObjectsData.Data.FloatFeatures.resize(featuresLayout->GetFloatFeatureCount());
                data->ObjectsData.Data.CatFeatures.resize(featuresLayout->GetCatFeatureCount());
                data->ObjectsData.Data.TextFeatures.resize(featuresLayout->GetTextFeatureCount());

                if (storeFeaturesDataAsExternalValuesHolders) {
                    // external columns keep the same subset
                    subsetIndexing = srcObjectsCommonData.SubsetIndexing;
                } else {
                    subsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                        TFullSubset<ui32>(objectsGrouping->GetObjectCount())
                    );
                }
            }

            {
                ui64 cpuRamUsage = NMemInfo::GetMemInfo().RSS;
                OutputWarningIfCpuRamUsageOverLimit(cpuRamUsage, options.CpuRamLimit);

                TResourceConstrainedExecutor resourceConstrainedExecutor(
                    "CPU RAM",
                    options.CpuRamLimit - Min(cpuRamUsage, options.CpuRamLimit),
                    true,
                    localExecutor
                );

                const bool calcBordersAndNanModeOnlyInProcessFloatFeatures =
                    calcBordersAndNanModeOnly || bundleExclusiveFeatures;

                featuresLayout->IterateOverAvailableFeatures<EFeatureType::Float>(
                    [&] (TFloatFeatureIdx floatFeatureIdx) {
                        // as pointer to capture in lambda
                        auto* srcFloatFeatureHolderPtr =
                            &(rawDataProvider->ObjectsData->Data.FloatFeatures[*floatFeatureIdx]);

                        resourceConstrainedExecutor.Add(
                            {
                                EstimateMemUsageForFloatFeature(
                                    **srcFloatFeatureHolderPtr,
                                    *quantizedFeaturesInfo,
                                    options,
                                    !calcBordersAndNanModeOnly,
                                    storeFeaturesDataAsExternalValuesHolders
                                ),
                                [&, floatFeatureIdx, srcFloatFeatureHolderPtr]() {
                                    ProcessFloatFeature(
                                        floatFeatureIdx,
                                        **srcFloatFeatureHolderPtr,
                                        subsetForBuildBorders ?
                                            subsetForBuildBorders.Get()
                                            : srcObjectsCommonData.SubsetIndexing.Get(),
                                        options,
                                        calcBordersAndNanModeOnlyInProcessFloatFeatures,
                                        storeFeaturesDataAsExternalValuesHolders,
                                        subsetIndexing.Get(),
                                        localExecutor,
                                        quantizedFeaturesInfo,
                                        calcBordersAndNanModeOnlyInProcessFloatFeatures ?
                                            nullptr
                                            : &(data->ObjectsData.Data.FloatFeatures[*floatFeatureIdx])
                                    );

                                    // exclusive features are bundled later by bundle,
                                    // binary features are binarized later by packs
                                    if (clearSrcObjectsData &&
                                        (calcBordersAndNanModeOnly ||
                                         (!bundleExclusiveFeatures &&
                                          !IsFloatFeatureToBeBinarized(
                                              options,
                                              *quantizedFeaturesInfo,
                                              floatFeatureIdx
                                         ))))
                                    {
                                        srcFloatFeatureHolderPtr->Destroy();
                                    }
                                }
                            }
                        );
                    }
                );

                if (!calcBordersAndNanModeOnly) {
                    const ui64 maxMemUsageForCatFeature = EstimateMaxMemUsageForCatFeature(
                        objectsGrouping->GetObjectCount(),
                        storeFeaturesDataAsExternalValuesHolders
                    );

                    featuresLayout->IterateOverAvailableFeatures<EFeatureType::Categorical>(
                         [&] (TCatFeatureIdx catFeatureIdx) {
                            resourceConstrainedExecutor.Add(
                                {
                                    maxMemUsageForCatFeature,
                                    [&, catFeatureIdx]() {
                                        auto& srcCatFeatureHolder =
                                            rawDataProvider->ObjectsData->Data.CatFeatures[*catFeatureIdx];

                                        ProcessCatFeature(
                                            catFeatureIdx,
                                            *srcCatFeatureHolder,
                                            /*updatePerfectHashOnly*/ bundleExclusiveFeatures,
                                            storeFeaturesDataAsExternalValuesHolders,
                                            /*mapMostFrequentValueTo0*/ bundleExclusiveFeatures,
                                            subsetIndexing.Get(),
                                            quantizedFeaturesInfo,
                                            &(data->ObjectsData.Data.CatFeatures[*catFeatureIdx])
                                        );

                                        // exclusive features are bundled later by bundle,
                                        // binary features are binarized later by packs
                                        if (clearSrcObjectsData &&
                                            (!bundleExclusiveFeatures &&
                                              !IsCatFeatureToBeBinarized(
                                                  options,
                                                  *quantizedFeaturesInfo,
                                                  catFeatureIdx
                                             )))
                                        {
                                            srcCatFeatureHolder.Destroy();
                                        }
                                    }
                                }
                            );
                        }
                    );


                    // tokenize text features
                    featuresLayout->IterateOverAvailableFeatures<EFeatureType::Text>(
                        [&] (TTextFeatureIdx textFeatureIdx) {
                            auto& srcTextFeatureHolder = rawDataProvider->ObjectsData->Data.TextFeatures[*textFeatureIdx];

                            ProcessTextFeature(
                                textFeatureIdx,
                                *srcTextFeatureHolder,
                                subsetIndexing.Get(),
                                quantizedFeaturesInfo,
                                &(data->ObjectsData.Data.TextFeatures[*textFeatureIdx])
                            );
                        }
                    );
                }

                resourceConstrainedExecutor.ExecTasks();
            }

            if (calcBordersAndNanModeOnly) {
                return nullptr;
            }

            CB_ENSURE(
                featuresLayout->HasAvailableAndNotIgnoredFeatures(),
                "All features are either constant or ignored."
            );

            data->ObjectsData.Data.QuantizedFeaturesInfo = quantizedFeaturesInfo;


            if (bundleExclusiveFeatures) {
                data->ObjectsData.ExclusiveFeatureBundlesData = TExclusiveFeatureBundlesData(
                    *(data->ObjectsData.Data.QuantizedFeaturesInfo),
                    CreateExclusiveFeatureBundles(
                        rawDataProvider->ObjectsData->Data,
                        *(srcObjectsCommonData.SubsetIndexing),
                        *(data->ObjectsData.Data.QuantizedFeaturesInfo),
                        options.ExclusiveFeaturesBundlingOptions,
                        localExecutor
                    )
                );
            }

            if (options.CpuCompatibleFormat && options.PackBinaryFeaturesForCpu) {
                data->ObjectsData.PackedBinaryFeaturesData = TPackedBinaryFeaturesData(
                    *data->ObjectsData.Data.QuantizedFeaturesInfo,
                    data->ObjectsData.ExclusiveFeatureBundlesData
                );
            }

            {
                ui64 cpuRamUsage = NMemInfo::GetMemInfo().RSS;
                OutputWarningIfCpuRamUsageOverLimit(cpuRamUsage, options.CpuRamLimit);

                TResourceConstrainedExecutor resourceConstrainedExecutor(
                    "CPU RAM",
                    options.CpuRamLimit - Min(cpuRamUsage, options.CpuRamLimit),
                    true,
                    localExecutor
                );

                if (bundleExclusiveFeatures) {
                    ScheduleBundleFeatures(
                        *(srcObjectsCommonData.SubsetIndexing),
                        clearSrcObjectsData,
                        subsetIndexing.Get(),
                        localExecutor,
                        &resourceConstrainedExecutor,
                        &rawDataProvider->ObjectsData->Data,
                        &data->ObjectsData
                    );

                    /*
                     * call it only if bundleExclusiveFeatures because otherwise they've already been
                     * created during Process(Float|Cat)Feature calls above
                     */
                    ScheduleNonBundledAndNonBinaryFeatures(
                        *(srcObjectsCommonData.SubsetIndexing),
                        clearSrcObjectsData,
                        subsetIndexing.Get(),
                        localExecutor,
                        &resourceConstrainedExecutor,
                        &rawDataProvider->ObjectsData->Data,
                        &data->ObjectsData
                    );
                }

                if (options.CpuCompatibleFormat && options.PackBinaryFeaturesForCpu) {
                    ScheduleBinarizeFeatures(
                        *(srcObjectsCommonData.SubsetIndexing),
                        clearSrcObjectsData,
                        subsetIndexing.Get(),
                        localExecutor,
                        &resourceConstrainedExecutor,
                        &rawDataProvider->ObjectsData->Data,
                        &data->ObjectsData
                    );
                }

                resourceConstrainedExecutor.ExecTasks();
            }

            if (clearSrcData) {
                data->MetaInfo = std::move(rawDataProvider->MetaInfo);
                data->TargetData = std::move(rawDataProvider->RawTargetData.Data);
                if (clearSrcObjectsData) {
                    data->CommonObjectsData = std::move(rawDataProvider->ObjectsData->CommonData);
                } else {
                    data->CommonObjectsData = rawDataProvider->ObjectsData->CommonData;
                }
            } else {
                data->MetaInfo = rawDataProvider->MetaInfo;
                data->TargetData = rawDataProvider->RawTargetData.Data;
                data->CommonObjectsData = rawDataProvider->ObjectsData->CommonData;
            }
            data->MetaInfo.FeaturesLayout = featuresLayout;
            data->CommonObjectsData.FeaturesLayout = featuresLayout;
            data->CommonObjectsData.SubsetIndexing = std::move(subsetIndexing);

            if (options.CpuCompatibleFormat) {
                return MakeDataProvider<TQuantizedForCPUObjectsDataProvider>(
                    objectsGrouping,
                    std::move(*data),
                    false,
                    localExecutor
                )->CastMoveTo<TQuantizedObjectsDataProvider>();
            } else {
                return MakeDataProvider<TQuantizedObjectsDataProvider>(
                    objectsGrouping,
                    CastToBase(std::move(*data)),
                    false,
                    localExecutor
                );
            }
        }
    };


    void CalcBordersAndNanMode(
        const TQuantizationOptions& options,
        TRawDataProviderPtr rawDataProvider,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    ) {
        TQuantizationImpl::Do(
            options,
            std::move(rawDataProvider),
            quantizedFeaturesInfo,
            true,
            rand,
            localExecutor
        );
    }

    TQuantizedObjectsDataProviderPtr Quantize(
        const TQuantizationOptions& options,
        TRawObjectsDataProviderPtr rawObjectsDataProvider,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    ) {
        TDataMetaInfo dataMetaInfo;
        dataMetaInfo.FeaturesLayout = rawObjectsDataProvider->GetFeaturesLayout();

        auto objectsGrouping = rawObjectsDataProvider->GetObjectsGrouping();

        TRawTargetData dummyData;
        dummyData.SetTrivialWeights(rawObjectsDataProvider->GetObjectCount());

        auto rawDataProvider = MakeIntrusive<TRawDataProvider>(
            std::move(dataMetaInfo),
            std::move(rawObjectsDataProvider),
            objectsGrouping,
            TRawTargetDataProvider(objectsGrouping, std::move(dummyData), true, nullptr)
        );

        auto quantizedDataProvider = Quantize(
            options,
            std::move(rawDataProvider),
            quantizedFeaturesInfo,
            rand,
            localExecutor
        );

        return quantizedDataProvider->ObjectsData;
    }


    TQuantizedDataProviderPtr Quantize(
        const TQuantizationOptions& options,
        TRawDataProviderPtr rawDataProvider,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    ) {
        return TQuantizationImpl::Do(
            options,
            std::move(rawDataProvider),
            quantizedFeaturesInfo,
            false,
            rand,
            localExecutor
        );
    }

    TQuantizedDataProviders Quantize(
        const TQuantizationOptions& options,
        const NCatboostOptions::TDataProcessingOptions& dataProcessingOptions,
        bool floatFeaturesAllowNansInTestOnly,
        TConstArrayRef<ui32> ignoredFeatures,
        TRawDataProviders rawDataProviders,
        TRestorableFastRng64* rand,
        NPar::TLocalExecutor* localExecutor
    ) {
        TQuantizedDataProviders result;
        auto quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *rawDataProviders.Learn->MetaInfo.FeaturesLayout,
            ignoredFeatures,
            dataProcessingOptions.FloatFeaturesBinarization.Get(),
            dataProcessingOptions.PerFloatFeatureQuantization.Get(),
            dataProcessingOptions.TextProcessing.Get(),
            floatFeaturesAllowNansInTestOnly,
            options.AllowWriteFiles
        );

        result.Learn = Quantize(
            options,
            std::move(rawDataProviders.Learn),
            quantizedFeaturesInfo,
            rand,
            localExecutor
        );

        // TODO(akhropov): quantize test data in parallel
        for (auto& rawTestData : rawDataProviders.Test) {
            result.Test.push_back(
                Quantize(options, std::move(rawTestData), quantizedFeaturesInfo, rand, localExecutor)
            );
        }

        return result;
    }

}
