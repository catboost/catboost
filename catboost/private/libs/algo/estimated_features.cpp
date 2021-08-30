#include "estimated_features.h"

#include <catboost/private/libs/quantization/utils.h>

#include <catboost/libs/data/columns.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/exception.h>

#include <library/cpp/grid_creator/binarization.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/system/types.h>


using namespace NCB;


static bool IsPackedBinaryFeatures(const TEstimatedFeaturesMeta& featuresMeta) {
    if (featuresMeta.UniqueValuesUpperBoundHint) {
        return AllOf(*featuresMeta.UniqueValuesUpperBoundHint, [](ui32 binCount) { return binCount == 2; });
    }
    return false;
}

template <class TFeatureEstimatorPtrType> // either TFeatureEstimator or TOnlineFeatureEstimator
static void CreateMetaData(
    bool isOnline,
    const NCatboostOptions::TBinarizationOptions& quantizationOptions,
    TConstArrayRef<TFeatureEstimatorPtrType> featureEstimatorsSubset,
    TQuantizedEstimatedFeaturesInfo* quantizedEstimatedFeaturesInfo,
    TVector<TMaybe<TPackedBinaryIndex>>* flatFeatureIndexToPackedBinaryIndex,
    TVector<TFeatureIdxWithType>* packedBinaryToSrcIndex,
    TVector<TEstimatedFeaturesMeta>* estimatedFeaturesMeta
) {
    auto& layout = quantizedEstimatedFeaturesInfo->Layout;
    for (size_t id : xrange(featureEstimatorsSubset.size())) {
        TEstimatorId estimatorId(SafeIntegerCast<ui32>(id), isOnline);

        estimatedFeaturesMeta->push_back(featureEstimatorsSubset[id]->FeaturesMeta());

        const ui32 estimatorFeatureCount = estimatedFeaturesMeta->back().FeaturesCount;

        if (IsPackedBinaryFeatures(estimatedFeaturesMeta->back())) {
            flatFeatureIndexToPackedBinaryIndex->resize(layout.size() + estimatorFeatureCount);

            for (auto localFeatureIdx : xrange(estimatorFeatureCount)) {
                const ui32 featureIdx = SafeIntegerCast<ui32>(layout.size());

                (*flatFeatureIndexToPackedBinaryIndex)[featureIdx]
                    = TPackedBinaryIndex::FromLinearIdx(packedBinaryToSrcIndex->size());
                packedBinaryToSrcIndex->push_back(TFeatureIdxWithType(EFeatureType::Float, featureIdx));

                layout.push_back(TEstimatedFeatureId{estimatorId, localFeatureIdx});
            }
        } else {
            for (auto localFeatureIdx : xrange(estimatorFeatureCount)) {
                layout.push_back(TEstimatedFeatureId{estimatorId, localFeatureIdx});
            }
        }
    }
    flatFeatureIndexToPackedBinaryIndex->resize(layout.size());

    if (!quantizedEstimatedFeaturesInfo->QuantizedFeaturesInfo) {
        quantizedEstimatedFeaturesInfo->QuantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            TFeaturesLayout(SafeIntegerCast<ui32>(layout.size())),
            /*ignoredFeatures*/ TConstArrayRef<ui32>(),
            quantizationOptions,
            /*perFloatFeatureQuantization*/ TMap<ui32, NCatboostOptions::TBinarizationOptions>(),
            /*floatFeaturesAllowNansInTestOnly*/ true
        );
    }
}


static void AppendBinaryFeatures(
    size_t dstEstimatorFeaturesOffset,
    TConstArrayRef<ui32> localFeatureIndices,
    TConstArrayRef<ui32> srcPackedData,
    TVector<TCompressedArray>* dstPackedData
) {
    constexpr size_t DST_FEATURES_PER_PACK = sizeof(TBinaryFeaturesPack) * CHAR_BIT;

    const size_t objectCount = srcPackedData.size();

    const size_t srcFeaturesCount = localFeatureIndices.size();
    const size_t dstCurrentFeatureCount = dstEstimatorFeaturesOffset + localFeatureIndices.front();
    const size_t dstNewFeatureCount = dstCurrentFeatureCount + srcFeaturesCount;
    const size_t dstPacksSize = CeilDiv(dstNewFeatureCount, DST_FEATURES_PER_PACK);

    ui32 srcFeaturesBegin = 0;
    ui32 dstFeaturesBegin = dstCurrentFeatureCount;

    for (size_t dstPackIdx = dstCurrentFeatureCount / DST_FEATURES_PER_PACK;
         dstPackIdx < dstPacksSize;
         ++dstPackIdx)
    {
        const size_t dstOffsetInPack = dstFeaturesBegin % DST_FEATURES_PER_PACK;
        const size_t featuresInPackCount = Min(
            DST_FEATURES_PER_PACK - dstOffsetInPack,
            srcFeaturesCount - srcFeaturesBegin
        );
        const ui32 srcMask = SafeIntegerCast<ui32>((1 << featuresInPackCount) - 1);

        if (!dstOffsetInPack) { // new pack
            (*dstPackedData).push_back(
                TCompressedArray::CreateWithUninitializedData(objectCount, DST_FEATURES_PER_PACK)
            );
            TArrayRef<TBinaryFeaturesPack> dstArray
                = (*dstPackedData).back().GetRawArray<TBinaryFeaturesPack>();
            Fill(dstArray.begin(), dstArray.end(), TBinaryFeaturesPack(0));
        }

        TBinaryFeaturesPack* dstPtr = (TBinaryFeaturesPack*)((*dstPackedData).back().GetRawPtr());

        for (auto objectIdx : xrange(objectCount)) {
            dstPtr[objectIdx] |= ((srcPackedData[objectIdx] >> srcFeaturesBegin) & srcMask) << dstOffsetInPack;
        }

        srcFeaturesBegin += featuresInPackCount;
        dstFeaturesBegin += featuresInPackCount;
    }
}

static TCalculatedFeatureVisitor CreatePackedFeatureWriter(
    ui32* featureCount,
    size_t* currentPackedFeatureCount,
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    TVector<TCompressedArray>* packedFeaturesData
) {
    return TCalculatedFeatureVisitor(
        [=](TConstArrayRef<ui32> localFeatureIndices, TConstArrayRef<ui32> packedValues) {
            for (auto localFeatureIdx : localFeatureIndices) {
                auto featureIdx = *featureCount + localFeatureIdx;
                if (!quantizedFeaturesInfo->HasBorders(TFloatFeatureIdx(featureIdx))) {
                    quantizedFeaturesInfo->SetBorders(TFloatFeatureIdx(featureIdx), TVector<float>{0.5f});
                }
            }

            AppendBinaryFeatures(
                *currentPackedFeatureCount,
                localFeatureIndices,
                packedValues,
                packedFeaturesData
            );
        }
    );
}


static void Quantize(
    ui32 featureIdx,
    TConstArrayRef<float> srcValues,
    TConstArrayRef<float> borders,
    const TArraySubsetIndexing<ui32>* fullSubsetIndexing,
    NPar::ILocalExecutor* localExecutor,
    THolder<IQuantizedFloatValuesHolder>* dstColumn
) {
    TTypeCastArraySubset<float, float> arraySubset(
        TMaybeOwningConstArrayHolder<float>::CreateNonOwning(srcValues),
        fullSubsetIndexing
    );

    const ui8 histogramWidth = CalcHistogramWidthForBorders(borders.size());

    TCompressedArray dstStorage = TCompressedArray::CreateWithUninitializedData(
        srcValues.size(),
        histogramWidth
    );

    auto quantize = [&] (auto dstArray) {
        NCB::Quantize(
            arraySubset,
            /*allowNans*/ false,
            ENanMode::Forbidden,
            featureIdx,
            borders,
            dstArray,
            localExecutor
        );
    };

    switch (histogramWidth) {
        case 8:
            quantize(dstStorage.GetRawArray<ui8>());
            break;
        case 16:
            quantize(dstStorage.GetRawArray<ui16>());
            break;
        default:
            CB_ENSURE_INTERNAL(false, "Unexpected " << LabeledOutput(histogramWidth));
    }

    *dstColumn = MakeHolder<TQuantizedFloatValuesHolder>(
        featureIdx,
        std::move(dstStorage),
        fullSubsetIndexing
    );
}


static TCalculatedFeatureVisitor CreateSingleFeatureWriter(
    ui32* featureCount,
    const TArraySubsetIndexing<ui32>* fullSubsetIndexing,
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    const TFeaturesArraySubsetIndexing* calcBordersSubset,
    NPar::ILocalExecutor* localExecutor,
    TVector<THolder<IQuantizedFloatValuesHolder>>* featuresData
) {
    return TCalculatedFeatureVisitor(
        [=](ui32 localFeatureIdx, TConstArrayRef<float> values) {
            const ui32 featureIdx = *featureCount + localFeatureIdx;

            if (!quantizedFeaturesInfo->HasBorders(TFloatFeatureIdx(featureIdx))) {
                TVector<float> valuesForQuantization = GetSubset<float>(
                    values,
                    *calcBordersSubset,
                    localExecutor
                );
                const NCatboostOptions::TBinarizationOptions& binarizationOptions
                    = quantizedFeaturesInfo->GetFloatFeatureBinarization(featureIdx);

                NSplitSelection::TQuantization quantization = NSplitSelection::BestSplit(
                    NSplitSelection::TFeatureValues(std::move(valuesForQuantization)),
                    /*featureValuesMayContainNans*/ false,
                    binarizationOptions.BorderCount.Get(),
                    binarizationOptions.BorderSelectionType,
                    /*quantizedDefaultBinFraction*/ Nothing(),
                    /*initialBorders*/ Nothing()
                );

                quantizedFeaturesInfo->SetBorders(
                    TFloatFeatureIdx(featureIdx),
                    std::move(quantization.Borders)
                );
            }

            Quantize(
                featureIdx,
                values,
                quantizedFeaturesInfo->GetBorders(TFloatFeatureIdx(featureIdx)),
                fullSubsetIndexing,
                localExecutor,
                &((*featuresData)[featureIdx])
            );
        }
    );
}


namespace {
    struct TCalculatedFeatureVisitors {
        TMaybe<TCalculatedFeatureVisitor> LearnVisitor; // TMaybe - to allow delayed initialization
        TVector<TCalculatedFeatureVisitor> TestVisitors;
    };
}


static TIntrusivePtr<TQuantizedObjectsDataProvider> CreateObjectsDataProvider(
    TObjectsGroupingPtr objectsGrouping,
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    TAtomicSharedPtr<TFeaturesArraySubsetIndexing> fullSubset,
    TConstArrayRef<TMaybe<TPackedBinaryIndex>> flatFeatureIndexToPackedBinaryIndex,
    TConstArrayRef<TFeatureIdxWithType> packedBinaryToSrcIndex,
    TVector<TCompressedArray>&& packedFeaturesData,
    TQuantizedObjectsData&& data
) {
    constexpr size_t BINARY_FEATURES_PER_PACK = sizeof(TBinaryFeaturesPack) * CHAR_BIT;

    TCommonObjectsData commonData;
    commonData.FeaturesLayout = quantizedFeaturesInfo->GetFeaturesLayout();
    commonData.SubsetIndexing = std::move(fullSubset);

    const size_t featureCount = commonData.FeaturesLayout->GetExternalFeatureCount();

    Y_ASSERT(flatFeatureIndexToPackedBinaryIndex.size() == featureCount);
    data.PackedBinaryFeaturesData.FlatFeatureIndexToPackedBinaryIndex.assign(
        flatFeatureIndexToPackedBinaryIndex.begin(),
        flatFeatureIndexToPackedBinaryIndex.end()
    );
    data.PackedBinaryFeaturesData.PackedBinaryToSrcIndex.assign(
        packedBinaryToSrcIndex.begin(),
        packedBinaryToSrcIndex.end()
    );
    data.PackedBinaryFeaturesData.SrcData.resize(packedFeaturesData.size());
    data.ExclusiveFeatureBundlesData.FlatFeatureIndexToBundlePart.resize(featureCount);
    data.FeaturesGroupsData.FlatFeatureIndexToGroupPart.resize(featureCount);

    const size_t binaryFeatureCount = packedBinaryToSrcIndex.size();

    size_t binaryFeatureIdx = 0;
    for (auto packIdx : xrange(packedFeaturesData.size())) {
        data.PackedBinaryFeaturesData.SrcData[packIdx] = MakeHolder<TBinaryPacksArrayHolder>(
            SafeIntegerCast<ui32>(packIdx),
            std::move(packedFeaturesData[packIdx]),
            commonData.SubsetIndexing.Get()
        );
        const size_t binaryFeatureIdxEnd = Min(
            binaryFeatureIdx + BINARY_FEATURES_PER_PACK,
            binaryFeatureCount
        );
        for (; binaryFeatureIdx < binaryFeatureIdxEnd; ++binaryFeatureIdx) {
            const ui32 featureIdx = packedBinaryToSrcIndex[binaryFeatureIdx].FeatureIdx;
            data.FloatFeatures[featureIdx] = MakeHolder<TQuantizedFloatPackedBinaryValuesHolder>(
                featureIdx,
                data.PackedBinaryFeaturesData.SrcData[packIdx].Get(),
                binaryFeatureIdx % BINARY_FEATURES_PER_PACK
            );
        }
    }

    data.QuantizedFeaturesInfo = quantizedFeaturesInfo;

    return MakeIntrusive<TQuantizedObjectsDataProvider>(
        std::move(objectsGrouping),
        std::move(commonData),
        std::move(data),
        /*skipCheck*/ true,
        /*localExecutor*/ Nothing()
    );
}


TEstimatedForCPUObjectsDataProviders NCB::CreateEstimatedFeaturesData(
    const NCatboostOptions::TBinarizationOptions& quantizationOptions,
    ui32 maxSubsetSizeForBuildBordersAlgorithms,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo, // can be nullptr
    const TTrainingDataProviders& trainingDataProviders,
    TFeatureEstimatorsPtr featureEstimators,
    TMaybe<TConstArrayRef<ui32>> learnPermutation,
    NPar::ILocalExecutor* localExecutor,
    TRestorableFastRng64* rand
) {
    const bool isOnline = learnPermutation.Defined();
    const size_t testCount = trainingDataProviders.Test.size();

    TEstimatedForCPUObjectsDataProviders result;
    result.QuantizedEstimatedFeaturesInfo.QuantizedFeaturesInfo = quantizedFeaturesInfo;

    TQuantizedObjectsData learnData;
    TVector<TQuantizedObjectsData> testData(testCount);

    // equal for both learn and test
    TVector<TMaybe<TPackedBinaryIndex>> flatFeatureIndexToPackedBinaryIndex; // [flatFeatureIdx]
    TVector<TFeatureIdxWithType> packedBinaryToSrcIndex; // [linearPackedBinaryIndex]

    TVector<TCompressedArray> learnPackedFeaturesData; // [packIdx]
    TVector<TVector<TCompressedArray>> testPackedFeaturesData(testCount); // [testIdx][packIdx]

    TVector<TEstimatedFeaturesMeta> estimatedFeaturesMeta;

    TConstArrayRef<TFeatureEstimatorPtr> featureEstimatorsSubset;
    TConstArrayRef<TOnlineFeatureEstimatorPtr> onlineFeatureEstimatorsSubset;

    auto createMetaData = [&] (auto featureEstimatorsArray) {
        CreateMetaData(
            isOnline,
            quantizationOptions,
            featureEstimatorsArray,
            &result.QuantizedEstimatedFeaturesInfo,
            &flatFeatureIndexToPackedBinaryIndex,
            &packedBinaryToSrcIndex,
            &estimatedFeaturesMeta
        );
    };

    if (isOnline) {
        onlineFeatureEstimatorsSubset = featureEstimators->GetOnlineFeatureEstimators();
        createMetaData(onlineFeatureEstimatorsSubset);
    } else {
        featureEstimatorsSubset = featureEstimators->GetOfflineFeatureEstimators();
        createMetaData(featureEstimatorsSubset);
    }

    quantizedFeaturesInfo = result.QuantizedEstimatedFeaturesInfo.QuantizedFeaturesInfo;

    TFeaturesArraySubsetIndexing learnCalcBordersSubset = GetArraySubsetForBuildBorders(
        trainingDataProviders.Learn->ObjectsData->GetObjectCount(),
        quantizationOptions.BorderSelectionType.Get(),
        trainingDataProviders.Learn->ObjectsData->GetOrder() == EObjectsOrder::RandomShuffled,
        maxSubsetSizeForBuildBordersAlgorithms,
        rand
    );


    const ui32 featureCount = quantizedFeaturesInfo->GetFeaturesLayout()->GetExternalFeatureCount();

    ui32 currentFeatureCount = 0;
    size_t currentPackedFeatureCount = 0;

    auto createSingleFeatureWriter = [&] (
        const TFeaturesArraySubsetIndexing* fullSubsetIndexing,
        TQuantizedObjectsData* data
    ) {
        return CreateSingleFeatureWriter(
            &currentFeatureCount,
            fullSubsetIndexing,
            quantizedFeaturesInfo,
            &learnCalcBordersSubset,
            localExecutor,
            &(data->FloatFeatures)
        );
    };

    auto createPackedFeatureWriter = [&] (TVector<TCompressedArray>* packedFeaturesData) {
        return CreatePackedFeatureWriter(
            &currentFeatureCount,
            &currentPackedFeatureCount,
            quantizedFeaturesInfo,
            packedFeaturesData
        );
    };

    TCalculatedFeatureVisitors singleVisitors;
    TCalculatedFeatureVisitors packedVisitors;

    TAtomicSharedPtr<TFeaturesArraySubsetIndexing> learnFullSubset
        = MakeAtomicShared<TFeaturesArraySubsetIndexing>(
            TFullSubset<ui32>(trainingDataProviders.Learn->GetObjectCount())
        );
    learnData.FloatFeatures.resize(featureCount);
    singleVisitors.LearnVisitor.ConstructInPlace(
        createSingleFeatureWriter(learnFullSubset.Get(), &learnData)
    );
    packedVisitors.LearnVisitor.ConstructInPlace(createPackedFeatureWriter(&learnPackedFeaturesData));

    TVector<TAtomicSharedPtr<TFeaturesArraySubsetIndexing>> testFullSubsets;

    for (auto testIdx : xrange(testCount)) {
        testFullSubsets.push_back(
            MakeAtomicShared<TFeaturesArraySubsetIndexing>(
                TFullSubset<ui32>(trainingDataProviders.Test[testIdx]->GetObjectCount())
            )
        );
        testData[testIdx].FloatFeatures.resize(featureCount);
        singleVisitors.TestVisitors.push_back(
            createSingleFeatureWriter(
                testFullSubsets.back().Get(),
                &testData[testIdx]
            )
        );
        packedVisitors.TestVisitors.push_back(createPackedFeatureWriter(&testPackedFeaturesData[testIdx]));
    }


    for (size_t id : xrange(estimatedFeaturesMeta.size())) {
        auto computeFeatures = [&] (TCalculatedFeatureVisitors visitors) {
            if (isOnline) {
                onlineFeatureEstimatorsSubset[id]->ComputeOnlineFeatures(
                    *learnPermutation,
                    std::move(*visitors.LearnVisitor),
                    visitors.TestVisitors,
                    localExecutor
                );
            } else {
                featureEstimatorsSubset[id]->ComputeFeatures(
                    std::move(*visitors.LearnVisitor),
                    visitors.TestVisitors,
                    localExecutor
                );
            }
        };

        if (IsPackedBinaryFeatures(estimatedFeaturesMeta[id])) {
            computeFeatures(packedVisitors);
            currentPackedFeatureCount += estimatedFeaturesMeta[id].FeaturesCount;
        } else {
            computeFeatures(singleVisitors);
        }
        currentFeatureCount += estimatedFeaturesMeta[id].FeaturesCount;
    }

    auto createObjectsDataProvider = [&] (
        TObjectsGroupingPtr objectsGrouping,
        TAtomicSharedPtr<TFeaturesArraySubsetIndexing> fullSubset,
        TVector<TCompressedArray>&& packedFeaturesData,
        TQuantizedObjectsData&& data
    ) {
        return CreateObjectsDataProvider(
            std::move(objectsGrouping),
            quantizedFeaturesInfo,
            std::move(fullSubset),
            flatFeatureIndexToPackedBinaryIndex,
            packedBinaryToSrcIndex,
            std::move(packedFeaturesData),
            std::move(data)
        );
    };

    result.Learn = createObjectsDataProvider(
        trainingDataProviders.Learn->ObjectsGrouping,
        std::move(learnFullSubset),
        std::move(learnPackedFeaturesData),
        std::move(learnData)
    );

    for (auto testIdx : xrange(testCount)) {
        result.Test.push_back(
            createObjectsDataProvider(
                trainingDataProviders.Test[testIdx]->ObjectsGrouping,
                std::move(testFullSubsets[testIdx]),
                std::move(testPackedFeaturesData[testIdx]),
                std::move(testData[testIdx])
            )
        );
    }
    result.FeatureEstimators = trainingDataProviders.FeatureEstimators;

    return result;
}
