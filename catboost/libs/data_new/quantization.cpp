#include "quantization.h"

#include "cat_feature_perfect_hash_helper.h"
#include "columns.h"
#include "external_columns.h"
#include "util.h"

#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/resource_constrained_executor.h>
#include <catboost/libs/logging/logging.h>
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

#include <functional>
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
                quantizedFeaturesInfo.GetFloatFeatureBinarization().BorderSelectionType,
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


    static ui64 EstimateMaxMemUsageForFloatFeature(
        ui32 objectCount,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TQuantizationOptions& options,
        bool doQuantization, // if false - only calc borders
        bool clearSrcData
    ) {
        ui64 result = 0;

        if (NeedToCalcBorders(quantizedFeaturesInfo)) {
            auto borderSelectionType =
                quantizedFeaturesInfo.GetFloatFeatureBinarization().BorderSelectionType;

            const ui32 sampleSize = GetSampleSizeForBorderSelectionType(
                objectCount,
                borderSelectionType,
                options.MaxSubsetSizeForSlowBuildBordersAlgorithms
            );

            result += sizeof(float) * sampleSize; // for copying to srcFeatureValuesForBuildBorders

            result += CalcMemoryForFindBestSplit(
                SafeIntegerCast<int>(quantizedFeaturesInfo.GetFloatFeatureBinarization().BorderCount.Get()),
                (size_t)sampleSize,
                borderSelectionType
            );
        }

        if (doQuantization && (options.CpuCompatibleFormat || clearSrcData)) {
            // for storing quantized data
            // TODO(akhropov): support other bitsPerKey. MLTOOLS-2425
            result += sizeof(ui8) * objectCount;
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
        const auto& binarizationOptions = quantizedFeaturesInfo.GetFloatFeatureBinarization();

        Y_VERIFY(binarizationOptions.BorderCount > 0);

        TMaybeOwningConstArraySubset<float, ui32> srcFeatureData = srcFeature.GetArrayData();

        TMaybeOwningConstArraySubset<float, ui32> srcDataForBuildBorders(
            srcFeatureData.GetSrc(),
            subsetForBuildBorders
        );

        // does not contain nans
        TVector<float> srcFeatureValuesForBuildBorders;
        srcFeatureValuesForBuildBorders.reserve(srcDataForBuildBorders.Size());

        bool hasNans = false;

        srcDataForBuildBorders.ForEach(
            [&] (ui32 /*idx*/, float value) {
                if (IsNan(value)) {
                    hasNans = true;
                } else {
                    srcFeatureValuesForBuildBorders.push_back(value);
                }
            }
        );

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

        Y_VERIFY(borders->size() < 256);
    }


    template <class TBase>
    static void SetBinaryFeatureColumn(
        ui32 featureId,
        TMaybeOwningArrayHolder<TBinaryFeaturesPack> binaryFeaturesPack,
        ui8 bitIdx,
        const TFeaturesArraySubsetIndexing* subsetIndexing,
        THolder<TBase>* featureColumn
    ) {
        featureColumn->Reset(
            new TPackedBinaryValuesHolderImpl<TBase>(
                featureId,
                std::move(binaryFeaturesPack),
                bitIdx,
                subsetIndexing
            )
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
        TConstArrayRef<float> srcRawData
            = **(rawObjectsData.FloatFeatures[*floatFeatureIdx]->GetArrayData().GetSrc());
        float border = quantizedObjectsData.QuantizedFeaturesInfo->GetBorders(floatFeatureIdx)[0];

        return [srcRawData, border](ui32 /*idx*/, ui32 srcIdx) -> TBinaryFeaturesPack {
            return srcRawData[srcIdx] >= border ?
                TBinaryFeaturesPack(1) : TBinaryFeaturesPack(0);
        };
    }

    static TGetBitFunction GetBinaryCatFeatureFunction(
        const TQuantizedObjectsData& quantizedObjectsData,
        TCatFeatureIdx catFeatureIdx
    ) {
        const ui32* nonpackedQuantizedValuesArrayBegin
            = *(dynamic_cast<const TQuantizedCatValuesHolder&>(
                    *quantizedObjectsData.CatFeatures[*catFeatureIdx]
                ).GetArrayData().GetSrc());

        return [nonpackedQuantizedValuesArrayBegin](ui32 idx, ui32 /*srcIdx*/) -> TBinaryFeaturesPack {
            Y_ASSERT(nonpackedQuantizedValuesArrayBegin[idx] < 2);
            return TBinaryFeaturesPack(nonpackedQuantizedValuesArrayBegin[idx]);
        };
    }


    static void BinarizeFeatures(
        const TFeaturesArraySubsetIndexing& rawDataSubsetIndexing,
        bool clearSrcObjectsData,
        const TFeaturesArraySubsetIndexing* quantizedDataSubsetIndexing,
        NPar::TLocalExecutor* localExecutor,
        TRawObjectsData* rawObjectsData,
        TQuantizedForCPUObjectsData* quantizedObjectsData
    ) {
        auto& packedBinaryFeaturesData = quantizedObjectsData->PackedBinaryFeaturesData;

        packedBinaryFeaturesData = TPackedBinaryFeaturesData(
            *quantizedObjectsData->Data.QuantizedFeaturesInfo
        );

        const auto& packedBinaryToSrcIndex = packedBinaryFeaturesData.PackedBinaryToSrcIndex;

        const ui32 objectCount = rawDataSubsetIndexing.Size();
        const size_t bitsPerPack = sizeof(TBinaryFeaturesPack) * CHAR_BIT;


        localExecutor->ExecRangeWithThrow(
            [&] (int packIdx) {
                TVector<TBinaryFeaturesPack> dstPackedFeaturesData;
                dstPackedFeaturesData.yresize(objectCount);

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
                    } else {
                        getBitFunctions.push_back(
                            GetBinaryCatFeatureFunction(quantizedObjectsData->Data, TCatFeatureIdx(it->second))
                        );
                    }
                }

                rawDataSubsetIndexing.ParallelForEach(
                    [&] (ui32 idx, ui32 srcIdx) {
                        TBinaryFeaturesPack pack = 0;
                        for (auto bitIdx : xrange(getBitFunctions.size())) {
                            pack |= (getBitFunctions[bitIdx](idx, srcIdx) << bitIdx);
                        }
                        dstPackedFeaturesData[idx] = pack;
                    },
                    localExecutor
                );

                packedBinaryFeaturesData.SrcData[packIdx]
                    = TMaybeOwningArrayHolder<TBinaryFeaturesPack>::CreateOwning(
                            std::move(dstPackedFeaturesData)
                        );

                for (ui8 bitIdx = 0; bitIdx < (endIdx - startIdx); ++bitIdx) {
                    auto it = packedBinaryToSrcIndex.begin() + startIdx + bitIdx;

                    if (it->first == EFeatureType::Float) {
                        SetBinaryFeatureColumn(
                            rawObjectsData->FloatFeatures[it->second]->GetId(),
                            packedBinaryFeaturesData.SrcData[packIdx],
                            bitIdx,
                            quantizedDataSubsetIndexing,
                            &(quantizedObjectsData->Data.FloatFeatures[it->second])
                        );
                        if (clearSrcObjectsData) {
                            rawObjectsData->FloatFeatures[it->second].Destroy();
                        }
                    } else {
                        SetBinaryFeatureColumn(
                            quantizedObjectsData->Data.CatFeatures[it->second]->GetId(),
                            packedBinaryFeaturesData.SrcData[packIdx],
                            bitIdx,
                            quantizedDataSubsetIndexing,
                            &(quantizedObjectsData->Data.CatFeatures[it->second])
                        );
                    }
                }
            },
            0,
            SafeIntegerCast<int>(packedBinaryFeaturesData.SrcData.size()),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }


    static void ProcessFloatFeature(
        TFloatFeatureIdx floatFeatureIdx,
        const TFloatValuesHolder& srcFeature,
        const TFeaturesArraySubsetIndexing* subsetForBuildBorders,
        const TQuantizationOptions& options,
        bool clearSrcData,
        bool calcBordersAndNanModeOnly,
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

        auto borderSelectionType =
            quantizedFeaturesInfo->GetFloatFeatureBinarization().BorderSelectionType;

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

        if (!calcBordersAndNanModeOnly && !borders.Empty()) {
            TMaybeOwningConstArraySubset<float, ui32> srcFeatureData = srcFeature.GetArrayData();

            if (!options.CpuCompatibleFormat && !clearSrcData) {
                // use GPU-only external columns
                *dstQuantizedFeature = MakeHolder<TExternalFloatValuesHolder>(
                    srcFeature.GetId(),
                    *srcFeatureData.GetSrc(),
                    dstSubsetIndexing,
                    quantizedFeaturesInfo
                );
            } else if (!options.CpuCompatibleFormat ||
                !options.PackBinaryFeaturesForCpu ||
                (borders.size() > 1)) // binary features are binarized later by packs
            {
                // TODO(akhropov): support other bitsPerKey. MLTOOLS-2425
                const ui32 bitsPerKey = 8;
                TIndexHelper<ui64> indexHelper(bitsPerKey);
                TVector<ui64> quantizedDataStorage;
                quantizedDataStorage.yresize(indexHelper.CompressedSize(srcFeatureData.Size()));

                TArrayRef<ui8> quantizedData(
                    reinterpret_cast<ui8*>(quantizedDataStorage.data()),
                    srcFeatureData.Size()
                );

                // it's ok even if it is learn data, for learn nans are checked at CalcBordersAndNanMode stage
                bool allowNans = (nanMode != ENanMode::Forbidden) ||
                    quantizedFeaturesInfo->GetFloatFeaturesAllowNansInTestOnly();

                Quantize(
                    srcFeatureData,
                    allowNans,
                    nanMode,
                    srcFeature.GetId(),
                    borders,
                    localExecutor,
                    &quantizedData
                );

                *dstQuantizedFeature = MakeHolder<TQuantizedFloatValuesHolder>(
                    srcFeature.GetId(),
                    TCompressedArray(
                        srcFeatureData.Size(),
                        indexHelper.GetBitsPerKey(),
                        TMaybeOwningArrayHolder<ui64>::CreateOwning(std::move(quantizedDataStorage))
                    ),
                    dstSubsetIndexing
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
        const TQuantizationOptions& options,
        bool clearSrcData
    ) {
        ui64 result = 0;

        constexpr ui32 ESTIMATED_FEATURES_PERFECT_HASH_MAP_NODE_SIZE = 32;

        // assuming worst-case that all values will be added to Features Perfect Hash as new.
        result += ESTIMATED_FEATURES_PERFECT_HASH_MAP_NODE_SIZE * objectCount;

        if (options.CpuCompatibleFormat || clearSrcData) {
            // for storing quantized data
            // TODO(akhropov): support other bitsPerKey. MLTOOLS-2425
            result += sizeof(ui32) * objectCount;
        }

        return result;
    }


    static void ProcessCatFeature(
        TCatFeatureIdx catFeatureIdx,
        const THashedCatValuesHolder& srcFeature,
        const TQuantizationOptions& options,
        bool clearSrcData,
        const TFeaturesArraySubsetIndexing* dstSubsetIndexing,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        THolder<IQuantizedCatValuesHolder>* dstQuantizedFeature
    ) {
        TMaybeOwningConstArraySubset<ui32, ui32> srcFeatureData = srcFeature.GetArrayData();

        // TODO(akhropov): support other bitsPerKey. MLTOOLS-2425
        const ui32 bitsPerKey = 32;
        TIndexHelper<ui64> indexHelper(bitsPerKey);
        TVector<ui64> quantizedDataStorage;
        TArrayRef<ui32> quantizedDataValue;

        // GPU-only external columns
        const bool storeAsExternalValuesHolder = !options.CpuCompatibleFormat && !clearSrcData;

        if (!storeAsExternalValuesHolder) {
            quantizedDataStorage.yresize(indexHelper.CompressedSize(srcFeatureData.Size()));
            quantizedDataValue = TArrayRef<ui32>(
                reinterpret_cast<ui32*>(quantizedDataStorage.data()),
                srcFeatureData.Size()
            );
        }

        {
            TCatFeaturesPerfectHashHelper catFeaturesPerfectHashHelper(quantizedFeaturesInfo);

            catFeaturesPerfectHashHelper.UpdatePerfectHashAndMaybeQuantize(
                catFeatureIdx,
                srcFeatureData,
                !storeAsExternalValuesHolder ? TMaybe<TArrayRef<ui32>*>(&quantizedDataValue) : Nothing()
            );
        }

        auto uniqueValuesCounts = quantizedFeaturesInfo->GetUniqueValuesCounts(catFeatureIdx);
        if (uniqueValuesCounts.OnLearnOnly > 1) {
            if (storeAsExternalValuesHolder) {
                *dstQuantizedFeature = MakeHolder<TExternalCatValuesHolder>(
                    srcFeature.GetId(),
                    *srcFeatureData.GetSrc(),
                    dstSubsetIndexing,
                    quantizedFeaturesInfo
                );
            } else {
                /* binary features are temporarily stored as TQuantizedCatValuesHolder
                 * and compressed to packs at the last stage of quantization processing
                 * then this dstQuantizedFeature will be replaced with TQuantizedCatPackedBinaryValuesHolder
                 */

                *dstQuantizedFeature = MakeHolder<TQuantizedCatValuesHolder>(
                    srcFeature.GetId(),
                    TCompressedArray(
                        srcFeatureData.Size(),
                        indexHelper.GetBitsPerKey(),
                        TMaybeOwningArrayHolder<ui64>::CreateOwning(std::move(quantizedDataStorage))
                    ),
                    dstSubsetIndexing
                );
            }
        } else {
            CATBOOST_DEBUG_LOG << "Categorical Feature #" << srcFeature.GetId() << " is constant" << Endl;

            quantizedFeaturesInfo->GetFeaturesLayout()->IgnoreExternalFeature(srcFeature.GetId());
        }
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

                data->ObjectsData.Data.FloatFeatures.resize(featuresLayout->GetFloatFeatureCount());
                data->ObjectsData.PackedBinaryFeaturesData.FloatFeatureToPackedBinaryIndex.resize(
                    featuresLayout->GetFloatFeatureCount()
                );

                data->ObjectsData.Data.CatFeatures.resize(featuresLayout->GetCatFeatureCount());
                data->ObjectsData.PackedBinaryFeaturesData.CatFeatureToPackedBinaryIndex.resize(
                    featuresLayout->GetCatFeatureCount()
                );

                subsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(
                    TFullSubset<ui32>(objectsGrouping->GetObjectCount())
                );
            }

            {
                ui64 cpuRamUsage = NMemInfo::GetMemInfo().RSS;
                OutputWarningIfCpuRamUsageOverLimit(cpuRamUsage, options.CpuRamLimit);

                TResourceConstrainedExecutor resourceConstrainedExecutor(
                    *localExecutor,
                    "CPU RAM",
                    options.CpuRamLimit - Min(cpuRamUsage, options.CpuRamLimit),
                    true
                );

                const ui64 maxMemUsageForFloatFeature = EstimateMaxMemUsageForFloatFeature(
                    objectsGrouping->GetObjectCount(),
                    *quantizedFeaturesInfo,
                    options,
                    !calcBordersAndNanModeOnly,
                    clearSrcObjectsData
                );

                featuresLayout->IterateOverAvailableFeatures<EFeatureType::Float>(
                    [&] (TFloatFeatureIdx floatFeatureIdx) {
                        resourceConstrainedExecutor.Add(
                            {
                                maxMemUsageForFloatFeature,
                                [&, floatFeatureIdx]() {
                                    auto& srcFloatFeatureHolder =
                                        rawDataProvider->ObjectsData->Data.FloatFeatures[*floatFeatureIdx];

                                    ProcessFloatFeature(
                                        floatFeatureIdx,
                                        *srcFloatFeatureHolder,
                                        subsetForBuildBorders ?
                                            subsetForBuildBorders.Get()
                                            : srcObjectsCommonData.SubsetIndexing.Get(),
                                        options,
                                        clearSrcObjectsData,
                                        calcBordersAndNanModeOnly,
                                        subsetIndexing.Get(),
                                        localExecutor,
                                        quantizedFeaturesInfo,
                                        calcBordersAndNanModeOnly ?
                                            nullptr
                                            : &(data->ObjectsData.Data.FloatFeatures[*floatFeatureIdx])
                                    );

                                    // binary features are binarized later by packs
                                    if (clearSrcObjectsData &&
                                        (calcBordersAndNanModeOnly ||
                                         !IsFloatFeatureToBeBinarized(
                                             options,
                                             *quantizedFeaturesInfo,
                                             floatFeatureIdx
                                         )))
                                    {
                                        srcFloatFeatureHolder.Destroy();
                                    }
                                }
                            }
                        );
                    }
                );

                if (!calcBordersAndNanModeOnly) {
                    const ui64 maxMemUsageForCatFeature = EstimateMaxMemUsageForCatFeature(
                        objectsGrouping->GetObjectCount(),
                        options,
                        clearSrcObjectsData
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
                                            options,
                                            clearSrcObjectsData,
                                            subsetIndexing.Get(),
                                            quantizedFeaturesInfo,
                                            &(data->ObjectsData.Data.CatFeatures[*catFeatureIdx])
                                        );

                                        /* binary features are binarized later by packs
                                         * but non-packed quantized data is still saved as an intermediate
                                         * in data->ObjectsData.Data.CatFeatures
                                         * so we can clear raw data anyway
                                         */
                                        if (clearSrcObjectsData) {
                                            srcCatFeatureHolder.Destroy();
                                        }
                                    }
                                }
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

            if (options.CpuCompatibleFormat && options.PackBinaryFeaturesForCpu) {
                BinarizeFeatures(
                    *(srcObjectsCommonData.SubsetIndexing),
                    clearSrcObjectsData,
                    subsetIndexing.Get(),
                    localExecutor,
                    &rawDataProvider->ObjectsData->Data,
                    &data->ObjectsData
                );
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
        const NCatboostOptions::TBinarizationOptions floatFeaturesBinarization,
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
            floatFeaturesBinarization,
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
