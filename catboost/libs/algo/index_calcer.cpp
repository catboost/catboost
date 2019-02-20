#include "index_calcer.h"

#include "features_data_helpers.h"
#include "score_calcer.h"
#include "online_ctr.h"
#include "tree_print.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data_new/model_dataset_compatibility.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <catboost/libs/helpers/dense_hash.h>

#include <library/containers/stack_vector/stack_vec.h>

#include <functional>


using namespace NCB;


static bool GetCtrSplit(const TSplit& split, int idxPermuted, const TOnlineCTR& ctr) {
    ui8 ctrValue = ctr.Feature[split.Ctr.CtrIdx]
                              [split.Ctr.TargetBorderIdx]
                              [split.Ctr.PriorIdx]
                              [idxPermuted];
    return ctrValue > split.BinBorder;
}

static inline ui8 GetFeatureSplitIdx(const TSplit& split) {
    return split.BinBorder;
}

static inline const ui8* GetFloatHistogram(
    const TSplit& split,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider) {

    return *(*objectsDataProvider.GetNonPackedFloatFeature((ui32)split.FeatureIdx))->GetArrayData().GetSrc();
}

static inline const ui32* GetRemappedCatFeatures(
    const TSplit& split,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider) {

    return *(*objectsDataProvider.GetNonPackedCatFeature((ui32)split.FeatureIdx))->GetArrayData().GetSrc();
}

template <typename TCount, typename TCmpOp, int vectorWidth>
inline void BuildIndicesKernel(
    const ui32* permutation,
    const TCount* histogram,
    TCmpOp cmpOp,
    int level,
    TIndexType* indices) {

    Y_ASSERT(vectorWidth == 4);
    const ui32 perm0 = permutation[0];
    const ui32 perm1 = permutation[1];
    const ui32 perm2 = permutation[2];
    const ui32 perm3 = permutation[3];
    const TCount hist0 = histogram[perm0];
    const TCount hist1 = histogram[perm1];
    const TCount hist2 = histogram[perm2];
    const TCount hist3 = histogram[perm3];
    const TIndexType idx0 = indices[0];
    const TIndexType idx1 = indices[1];
    const TIndexType idx2 = indices[2];
    const TIndexType idx3 = indices[3];
    indices[0] = idx0 + cmpOp(hist0) * level;
    indices[1] = idx1 + cmpOp(hist1) * level;
    indices[2] = idx2 + cmpOp(hist2) * level;
    indices[3] = idx3 + cmpOp(hist3) * level;
}


template <typename TCount, typename TCmpOp>
inline void OfflineCtrBlock(
    const NPar::TLocalExecutor::TExecRangeParams& params,
    int blockIdx,
    const ui32* permutation,
    const TCount* histogram,
    TCmpOp cmpOp,
    int level,
    TIndexType* indices) {

    const int blockStart = blockIdx * params.GetBlockSize();
    const int nextBlockStart = Min<ui64>(blockStart + params.GetBlockSize(), params.LastId);
    constexpr int vectorWidth = 4;
    int doc;
    for (doc = blockStart; doc + vectorWidth <= nextBlockStart; doc += vectorWidth) {
        BuildIndicesKernel<TCount, TCmpOp, vectorWidth>(permutation + doc, histogram, cmpOp, level, indices + doc);
    }
    for (; doc < nextBlockStart; ++doc) {
        const int idxOriginal = permutation[doc];
        indices[doc] += cmpOp(histogram[idxOriginal]) * level;
    }
}

template <typename TCount, class TCmpOp>
inline void OfflineCtrBlock(
    const NPar::TLocalExecutor::TExecRangeParams& params,
    int blockIdx,
    TMaybe<TPackedBinaryIndex> maybeBinaryIndex,
    const ui32* permutation,
    const TCount* histogram, // can be nullptr if maybeBinaryIndex
    std::function<TPackedBinaryFeaturesArraySubset(ui32)>&& getBinaryFeaturesPack,
    TCmpOp cmpOp,
    int level,
    TIndexType* indices) {

    if (maybeBinaryIndex) {
        TBinaryFeaturesPack bitMask = TBinaryFeaturesPack(1) << maybeBinaryIndex->BitIdx;
        TBinaryFeaturesPack bitIdx = maybeBinaryIndex->BitIdx;

        NCB::TPackedBinaryFeaturesArraySubset packSubset = getBinaryFeaturesPack(maybeBinaryIndex->PackIdx);

        OfflineCtrBlock(
            params,
            blockIdx,
            permutation,
            (**packSubset.GetSrc()).data(),
            [bitMask, bitIdx, cmpOp = std::move(cmpOp)] (NCB::TBinaryFeaturesPack featuresPack) {
                return cmpOp((featuresPack & bitMask) >> bitIdx);
            },
            level,
            indices);
    } else {
        OfflineCtrBlock(
            params,
            blockIdx,
            permutation,
            histogram,
            std::move(cmpOp),
            level,
            indices);
    }
}


void SetPermutedIndices(
    const TSplit& split,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    int curDepth,
    const TFold& fold,
    TVector<TIndexType>* indices,
    NPar::TLocalExecutor* localExecutor) {

    CB_ENSURE(curDepth > 0);

    const int blockSize = 1000;
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, indices->ysize());
    blockParams.SetBlockSize(blockSize);

    const int splitWeight = 1 << (curDepth - 1);
    TIndexType* indicesData = indices->data();
    if (split.Type == ESplitType::FloatFeature) {
        auto floatFeatureIdx = TFloatFeatureIdx((ui32)split.FeatureIdx);

        const ui8* histogram = nullptr;
        auto maybeBinaryIndex = objectsDataProvider.GetFloatFeatureToPackedBinaryIndex(floatFeatureIdx);
        if (!maybeBinaryIndex) {
            histogram = GetFloatHistogram(split, objectsDataProvider);
        }

        localExecutor->ExecRange(
            [&](int blockIdx) {
                OfflineCtrBlock(
                    blockParams,
                    blockIdx,
                    maybeBinaryIndex,
                    fold.LearnPermutationFeaturesSubset.Get<TIndexedSubset<ui32>>().data(),
                    histogram,
                    [&] (ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                    [splitIdx = GetFeatureSplitIdx(split)] (ui8 bucket) {
                        return IsTrueHistogram(bucket, splitIdx);
                    },
                    splitWeight,
                    indicesData);
            },
            0,
            blockParams.GetBlockCount(),
            NPar::TLocalExecutor::WAIT_COMPLETE);
    } else if (split.Type == ESplitType::OnlineCtr) {
        auto& ctr = fold.GetCtr(split.Ctr.Projection);
        localExecutor->ExecRange(
            [&] (int i) {
                indicesData[i] += GetCtrSplit(split, i, ctr) * splitWeight;
            },
            blockParams,
            NPar::TLocalExecutor::WAIT_COMPLETE);
    } else {
        Y_ASSERT(split.Type == ESplitType::OneHotFeature);

        auto catFeatureIdx = TCatFeatureIdx((ui32)split.FeatureIdx);

        const ui32* histogram = nullptr;
        auto maybeBinaryIndex = objectsDataProvider.GetCatFeatureToPackedBinaryIndex(catFeatureIdx);
        if (!maybeBinaryIndex) {
            histogram = GetRemappedCatFeatures(split, objectsDataProvider);
        }

        localExecutor->ExecRange(
            [&] (int blockIdx) {
                OfflineCtrBlock(
                    blockParams,
                    blockIdx,
                    maybeBinaryIndex,
                    fold.LearnPermutationFeaturesSubset.Get<TIndexedSubset<ui32>>().data(),
                    histogram,
                    [&] (ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                    [bucketIdx = (ui32)split.BinBorder] (ui32 bucket) {
                        return IsTrueOneHotFeature(bucket, bucketIdx);
                    },
                    splitWeight,
                    indicesData);
            },
            0,
            blockParams.GetBlockCount(),
            NPar::TLocalExecutor::WAIT_COMPLETE);
    }
}

TVector<bool> GetIsLeafEmpty(int curDepth, const TVector<TIndexType>& indices) {
    TVector<bool> isLeafEmpty(1 << curDepth, true);
    for (const auto& idx : indices) {
        isLeafEmpty[idx] = false;
    }
    return isLeafEmpty;
}

int GetRedundantSplitIdx(const TVector<bool>& isLeafEmpty) {
    const int leafCount = isLeafEmpty.ysize();
    for (int splitIdx = 0; (1 << splitIdx) < leafCount; ++splitIdx) {
        bool isRedundantSplit = true;
        for (int idx = 0; idx < leafCount; ++idx) {
            if (idx & (1 << splitIdx)) {
                continue;
            }
            if (!isLeafEmpty[idx] && !isLeafEmpty[idx ^ (1 << splitIdx)]) {
                isRedundantSplit = false;
                break;
            }
        }
        if (isRedundantSplit) {
            return splitIdx;
        }
    }

    return -1;
}

// Get OnlineCTRs associated with a fold
static TVector<const TOnlineCTR*> GetOnlineCtrs(const TFold& fold, const TSplitTree& tree) {
    TVector<const TOnlineCTR*> onlineCtrs(tree.GetDepth());
    for (int splitIdx = 0; splitIdx < tree.GetDepth(); ++splitIdx) {
        const auto& split = tree.Splits[splitIdx];
        if (split.Type == ESplitType::OnlineCtr) {
            onlineCtrs[splitIdx] = &fold.GetCtr(split.Ctr.Projection);
        }
    }
    return onlineCtrs;
}

static void BuildIndicesForDataset(
    const TSplitTree& tree,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const NCB::TFeaturesArraySubsetIndexing& featuresArraySubsetIndexing,
    ui32 sampleCount,
    const TVector<const TOnlineCTR*>& onlineCtrs,
    int docOffset,
    NPar::TLocalExecutor* localExecutor,
    TIndexType* indices) {

    const ui32* permutation = nullptr;
    TVector<ui32> permutationStorage;
    if (HoldsAlternative<TIndexedSubset<ui32>>(featuresArraySubsetIndexing)) {
        permutation = featuresArraySubsetIndexing.Get<TIndexedSubset<ui32>>().data();
    } else {
        permutationStorage.yresize(featuresArraySubsetIndexing.Size());
        featuresArraySubsetIndexing.ParallelForEach(
            [&](ui32 idx, ui32 srcIdx) { permutationStorage[idx] = srcIdx; },
            localExecutor
        );
        permutation = permutationStorage.data();
    }

    const int blockSize = 1000;
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, (int)sampleCount);
    blockParams.SetBlockSize(blockSize);

    // precalc to avoid recalculation in each block
    TStackVec<const ui8*> splitFloatHistograms;
    splitFloatHistograms.yresize(tree.GetDepth());

    TStackVec<const ui32*> splitRemappedCatHistograms;
    splitRemappedCatHistograms.yresize(tree.GetDepth());

    for (auto splitIdx : xrange(tree.GetDepth())) {
        const auto& split = tree.Splits[splitIdx];
        if (split.Type == ESplitType::FloatFeature) {
            if (!objectsDataProvider.IsFeaturePackedBinary(TFloatFeatureIdx((ui32)split.FeatureIdx))) {
                splitFloatHistograms[splitIdx] = GetFloatHistogram(split, objectsDataProvider);
            }
        } else if (split.Type == ESplitType::OneHotFeature) {
            if (!objectsDataProvider.IsFeaturePackedBinary(TCatFeatureIdx((ui32)split.FeatureIdx))) {
                splitRemappedCatHistograms[splitIdx] = GetRemappedCatFeatures(split, objectsDataProvider);
            }
        }
    }


    auto updateLearnIndex = [&](int blockIdx) {
        for (int splitIdx = 0; splitIdx < tree.GetDepth(); ++splitIdx) {
            const auto& split = tree.Splits[splitIdx];
            const int splitWeight = 1 << splitIdx;
            if (split.Type == ESplitType::FloatFeature) {
                auto floatFeatureIdx = TFloatFeatureIdx((ui32)split.FeatureIdx);

                OfflineCtrBlock(
                    blockParams,
                    blockIdx,
                    objectsDataProvider.GetFloatFeatureToPackedBinaryIndex(floatFeatureIdx),
                    permutation,
                    splitFloatHistograms[splitIdx],
                    [&] (ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                    [splitIdx = GetFeatureSplitIdx(split)] (ui8 bucket) {
                        return IsTrueHistogram(bucket, splitIdx);
                    },
                    splitWeight,
                    indices);
            } else if (split.Type == ESplitType::OnlineCtr) {
                const TOnlineCTR& splitOnlineCtr = *onlineCtrs[splitIdx];
                NPar::TLocalExecutor::BlockedLoopBody(
                    blockParams,
                    [&](int doc) {
                        indices[doc] += GetCtrSplit(split, doc + docOffset, splitOnlineCtr) * splitWeight;
                    }
                )(blockIdx);
            } else {
                Y_ASSERT(split.Type == ESplitType::OneHotFeature);

                auto catFeatureIdx = TCatFeatureIdx((ui32)split.FeatureIdx);

                OfflineCtrBlock(
                    blockParams,
                    blockIdx,
                    objectsDataProvider.GetCatFeatureToPackedBinaryIndex(catFeatureIdx),
                    permutation,
                    splitRemappedCatHistograms[splitIdx],
                    [&] (ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                    [bucketIdx = (ui32)split.BinBorder] (ui32 bucket) {
                        return IsTrueOneHotFeature(bucket, bucketIdx);
                    },
                    splitWeight,
                    indices);
            }
        }
    };

    localExecutor->ExecRange(
        updateLearnIndex,
        0,
        blockParams.GetBlockCount(),
        NPar::TLocalExecutor::WAIT_COMPLETE);
}

TVector<TIndexType> BuildIndices(
    const TFold& fold,
    const TSplitTree& tree,
    NCB::TTrainingForCPUDataProviderPtr learnData, // can be nullptr
    TConstArrayRef<NCB::TTrainingForCPUDataProviderPtr> testData, // can be empty
    NPar::TLocalExecutor* localExecutor) {

    ui32 learnSampleCount = learnData ? learnData->GetObjectCount() : 0;
    ui32 tailSampleCount = 0;
    for (const auto& testSet : testData) {
        tailSampleCount += testSet->GetObjectCount();
    }

    const TVector<const TOnlineCTR*>& onlineCtrs = GetOnlineCtrs(fold, tree);

    TVector<TIndexType> indices(learnSampleCount + tailSampleCount);

    if (learnData) {
        BuildIndicesForDataset(
            tree,
            *learnData->ObjectsData,
            fold.LearnPermutationFeaturesSubset,
            learnSampleCount,
            onlineCtrs,
            0,
            localExecutor,
            indices.begin());
    }
    ui32 docOffset = learnSampleCount;
    for (size_t testIdx = 0; testIdx < testData.size(); ++testIdx) {
        const auto& testSet = *testData[testIdx];
        BuildIndicesForDataset(
            tree,
            *testSet.ObjectsData,
            testSet.ObjectsData->GetFeaturesArraySubsetIndexing(),
            testSet.GetObjectCount(),
            onlineCtrs,
            (int)docOffset,
            localExecutor,
            indices.begin() + docOffset);
        docOffset += testSet.GetObjectCount();
    }
    return indices;
}

static void BinarizeRawFeatures(
    const TFullModel& model,
    const NCB::TRawObjectsDataProvider& rawObjectsData,
    size_t start,
    size_t end,
    TVector<ui8>* result) {

    THashMap<ui32, ui32> columnReorderMap;
    CheckModelAndDatasetCompatibility(model, rawObjectsData, &columnReorderMap);
    auto docCount = end - start;
    result->resize(model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount() * docCount);
    TVector<ui32> transposedHash(docCount * model.GetUsedCatFeaturesCount());
    TVector<float> ctrs(model.ObliviousTrees.GetUsedModelCtrs().size() * docCount);

    const ui32 consecutiveSubsetBegin = GetConsecutiveSubsetBegin(rawObjectsData);
    const auto& featuresLayout = *rawObjectsData.GetFeaturesLayout();

    auto getFeatureDataBeginPtr = [&](ui32 flatFeatureIdx, TVector<TMaybe<NCB::TPackedBinaryIndex>>*) -> const float* {
        return GetRawFeatureDataBeginPtr(
            rawObjectsData,
            consecutiveSubsetBegin,
            flatFeatureIdx);
    };

    TVector<TConstArrayRef<float>> repackedFeatures;
    GetRepackedFeatures(
        start,
        end,
        model.ObliviousTrees.GetFlatFeatureVectorExpectedSize(),
        columnReorderMap,
        getFeatureDataBeginPtr,
        featuresLayout,
        &repackedFeatures);

    BinarizeFeatures(model,
        [&repackedFeatures](const TFloatFeature& floatFeature, size_t index) -> float {
            return repackedFeatures[floatFeature.FlatFeatureIndex][index];
        },
        [&repackedFeatures](const TCatFeature& catFeature, size_t index) -> ui32 {
            return ConvertFloatCatFeatureToIntHash(repackedFeatures[catFeature.FlatFeatureIndex][index]);
        },
        0,
        docCount,
        *result,
        transposedHash,
        ctrs);
}

static void AssignFeatureBins(
    const TFullModel& model,
    const NCB::TQuantizedForCPUObjectsDataProvider& quantizedObjectsData,
    size_t start,
    size_t end,
    TVector<ui8>* result)
{
    THashMap<ui32, ui32> columnReorderMap;
    CheckModelAndDatasetCompatibility(model, quantizedObjectsData, &columnReorderMap);
    auto docCount = end - start;
    result->resize(model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount() * docCount);
    auto floatBinsRemap = GetFloatFeaturesBordersRemap(model, *quantizedObjectsData.GetQuantizedFeaturesInfo().Get());
    const ui32 consecutiveSubsetBegin = NCB::GetConsecutiveSubsetBegin(quantizedObjectsData);

    auto getFeatureDataBeginPtr = [&](ui32 featureIdx, TVector<TMaybe<NCB::TPackedBinaryIndex>>* packedIdx) -> const ui8* {
        (*packedIdx)[featureIdx] = quantizedObjectsData.GetFloatFeatureToPackedBinaryIndex(TFeatureIdx<EFeatureType::Float>(featureIdx));
        if (!(*packedIdx)[featureIdx].Defined()) {
            return GetQuantizedForCpuFloatFeatureDataBeginPtr(
                    quantizedObjectsData,
                    consecutiveSubsetBegin,
                    featureIdx);
        } else {
            return (**quantizedObjectsData.GetBinaryFeaturesPack((*packedIdx)[featureIdx]->PackIdx).GetSrc()).Data();
        }
    };
    TVector<TConstArrayRef<ui8>> repackedBinFeatures;
    TVector<TMaybe<TPackedBinaryIndex>> packedIndexes;
    GetRepackedFeatures(
        start,
        end,
        model.ObliviousTrees.GetFlatFeatureVectorExpectedSize(),
        columnReorderMap,
        getFeatureDataBeginPtr,
        *quantizedObjectsData.GetFeaturesLayout().Get(),
        &repackedBinFeatures,
        &packedIndexes);

    AssignFeatureBins(
        model,
        [&floatBinsRemap, &repackedBinFeatures, &packedIndexes](const TFloatFeature& floatFeature, size_t index) -> ui8 {
            return QuantizedFeaturesFloatAccessor(floatBinsRemap, repackedBinFeatures, packedIndexes, floatFeature, index);
        },
        nullptr,
        0,
        end - start,
        *result);
}

TVector<ui8> GetModelCompatibleQuantizedFeatures(
    const TFullModel& model,
    const NCB::TObjectsDataProvider& objectsData,
    size_t start,
    size_t end)
{
    TVector<ui8> result;
    if (const auto* const rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(&objectsData)) {
        BinarizeRawFeatures(model, *rawObjectsData, start, end, &result);
    } else if (
        const auto* const quantizedObjectsData = dynamic_cast<const TQuantizedForCPUObjectsDataProvider*>(&objectsData))
    {
        AssignFeatureBins(model, *quantizedObjectsData, start, end, &result);
    } else {
        ythrow TCatBoostException() << "Unsupported objects data - neither raw nor quantized for CPU";
    }
    return result;
}

TVector<ui8> GetModelCompatibleQuantizedFeatures(const TFullModel& model, const NCB::TObjectsDataProvider& objectsData) {
    return GetModelCompatibleQuantizedFeatures(model, objectsData, /*start*/0, objectsData.GetObjectCount());
}

TVector<TIndexType> BuildIndicesForBinTree(
    const TFullModel& model,
    const TVector<ui8>& binarizedFeatures,
    size_t treeId) {

    if (model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount() == 0) {
        return TVector<TIndexType>();
    }

    auto docCount = binarizedFeatures.size() / model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount();
    TVector<TIndexType> indexesVec(docCount);
    const auto* treeSplitsCurPtr = model.ObliviousTrees.GetRepackedBins().data()
        + model.ObliviousTrees.TreeStartOffsets[treeId];
    CalcIndexes(
        !model.ObliviousTrees.OneHotFeatures.empty(),
        binarizedFeatures.data(),
        docCount,
        indexesVec.data(),
        treeSplitsCurPtr,
        model.ObliviousTrees.TreeSizes[treeId]);
    return indexesVec;
}

const ui8* GetFeatureDataBeginPtr(
    const NCB::TQuantizedForCPUObjectsDataProvider& quantizedObjectsData,
    ui32 featureIdx,
    int consecutiveSubsetBegin,
    TVector<TMaybe<NCB::TPackedBinaryIndex>>* packedIdx)
{
    (*packedIdx)[featureIdx] = quantizedObjectsData.GetFloatFeatureToPackedBinaryIndex(NCB::TFeatureIdx<EFeatureType::Float>(featureIdx));
    if (!(*packedIdx)[featureIdx].Defined()) {
        return GetQuantizedForCpuFloatFeatureDataBeginPtr(
            quantizedObjectsData,
            consecutiveSubsetBegin,
            featureIdx);
    } else {
        return (**quantizedObjectsData.GetBinaryFeaturesPack((*packedIdx)[featureIdx]->PackIdx).GetSrc()).Data();
    }
}
