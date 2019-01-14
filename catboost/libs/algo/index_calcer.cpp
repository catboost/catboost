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


using namespace NCB;


static bool GetCtrSplit(const TSplit& split, int idxPermuted,
                        const TOnlineCTR& ctr) {

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
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider
) {
    return *(*objectsDataProvider.GetFloatFeature((ui32)split.FeatureIdx))->GetArrayData().GetSrc();
}

static inline const ui32* GetRemappedCatFeatures(
    const TSplit& split,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider
) {
    return *(*objectsDataProvider.GetCatFeature((ui32)split.FeatureIdx))->GetArrayData().GetSrc();
}

template <typename TCount, bool (*CmpOp)(TCount, TCount), int vectorWidth>
void BuildIndicesKernel(const ui32* permutation, const TCount* histogram, TCount value, int level, TIndexType* indices) {
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
    indices[0] = idx0 + CmpOp(hist0, value) * level;
    indices[1] = idx1 + CmpOp(hist1, value) * level;
    indices[2] = idx2 + CmpOp(hist2, value) * level;
    indices[3] = idx3 + CmpOp(hist3, value) * level;
}

template <typename TCount, bool (*CmpOp)(TCount, TCount)>
void OfflineCtrBlock(const NPar::TLocalExecutor::TExecRangeParams& params,
                     int blockIdx,
                     const ui32* permutation,
                     const TCount* histogram,
                     TCount value,
                     int level,
                     TIndexType* indices) {
    const int blockStart = blockIdx * params.GetBlockSize();
    const int nextBlockStart = Min<ui64>(blockStart + params.GetBlockSize(), params.LastId);
    constexpr int vectorWidth = 4;
    int doc;
    for (doc = blockStart; doc + vectorWidth <= nextBlockStart; doc += vectorWidth) {
        BuildIndicesKernel<TCount, CmpOp, vectorWidth>(permutation + doc, histogram, value, level, indices + doc);
    }
    for (; doc < nextBlockStart; ++doc) {
        const int idxOriginal = permutation[doc];
        indices[doc] += CmpOp(histogram[idxOriginal], value) * level;
    }
}

void SetPermutedIndices(const TSplit& split,
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
        localExecutor->ExecRange([&](int blockIdx) {
            OfflineCtrBlock<ui8, IsTrueHistogram>(blockParams, blockIdx,
                fold.LearnPermutationFeaturesSubset.Get<TIndexedSubset<ui32>>().data(),
                GetFloatHistogram(split, objectsDataProvider),
                GetFeatureSplitIdx(split), splitWeight, indicesData);
        }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
    } else if (split.Type == ESplitType::OnlineCtr) {
        auto& ctr = fold.GetCtr(split.Ctr.Projection);
        localExecutor->ExecRange([&] (int i) {
            indicesData[i] += GetCtrSplit(split, i, ctr) * splitWeight;
        }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
    } else {
        Y_ASSERT(split.Type == ESplitType::OneHotFeature);
        localExecutor->ExecRange([&] (int blockIdx) {
            OfflineCtrBlock<ui32, IsTrueOneHotFeature>(blockParams, blockIdx,
                fold.LearnPermutationFeaturesSubset.Get<TIndexedSubset<ui32>>().data(),
                GetRemappedCatFeatures(split, objectsDataProvider),
                (ui32)split.BinBorder, splitWeight, indicesData);
        }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
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

static void BuildIndicesForDataset(const TSplitTree& tree,
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

    auto updateLearnIndex = [&](int blockIdx) {
        for (int splitIdx = 0; splitIdx < tree.GetDepth(); ++splitIdx) {
            const auto& split = tree.Splits[splitIdx];
            const int splitWeight = 1 << splitIdx;
            if (split.Type == ESplitType::FloatFeature) {
                OfflineCtrBlock<ui8, IsTrueHistogram>(blockParams, blockIdx, permutation,
                    GetFloatHistogram(split, objectsDataProvider),
                    GetFeatureSplitIdx(split), splitWeight, indices);
            } else if (split.Type == ESplitType::OnlineCtr) {
                const TOnlineCTR& splitOnlineCtr = *onlineCtrs[splitIdx];
                NPar::TLocalExecutor::BlockedLoopBody(blockParams, [&](int doc) {
                    indices[doc] += GetCtrSplit(split, doc + docOffset, splitOnlineCtr) * splitWeight;
                })(blockIdx);
            } else {
                Y_ASSERT(split.Type == ESplitType::OneHotFeature);
                OfflineCtrBlock<ui32, IsTrueOneHotFeature>(blockParams, blockIdx, permutation,
                    GetRemappedCatFeatures(split, objectsDataProvider),
                    (ui32)split.BinBorder, splitWeight, indices);
            }
        }
    };

    localExecutor->ExecRange(updateLearnIndex, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

TVector<TIndexType> BuildIndices(const TFold& fold,
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

void BinarizeFeatures(const TFullModel& model,
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
    const ui32 flatFeaturesCount = featuresLayout.GetExternalFeatureCount();

    auto getFeatureDataBeginPtr = [&](ui32 flatFeatureIdx) -> const float* {
        return GetRawFeatureDataBeginPtr(
            rawObjectsData,
            featuresLayout,
            consecutiveSubsetBegin,
            flatFeatureIdx);
    };

    TVector<TConstArrayRef<float>> repackedFeatures(model.ObliviousTrees.GetFlatFeatureVectorExpectedSize());
    if (columnReorderMap.empty()) {
        for (ui32 i = 0; i < flatFeaturesCount; ++i) {
            repackedFeatures[i] = MakeArrayRef(getFeatureDataBeginPtr(i) + start, docCount);
        }
    } else {
        for (const auto& [origIdx, sourceIdx] : columnReorderMap) {
            repackedFeatures[origIdx] = MakeArrayRef(getFeatureDataBeginPtr(sourceIdx) + start, docCount);
        }
    }

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

TVector<ui8> BinarizeFeatures(const TFullModel& model,
                              const NCB::TRawObjectsDataProvider& rawObjectsData,
                              size_t start,
                              size_t end) {
    TVector<ui8> result;
    BinarizeFeatures(model, rawObjectsData, start, end, &result);
    return result;
}

TVector<ui8> BinarizeFeatures(const TFullModel& model, const NCB::TRawObjectsDataProvider& rawObjectsData) {
    return BinarizeFeatures(model, rawObjectsData, /*start*/0, rawObjectsData.GetObjectCount());
}

TVector<TIndexType> BuildIndicesForBinTree(const TFullModel& model, const TVector<ui8>& binarizedFeatures, size_t treeId) {
    if (model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount() == 0) {
        return TVector<TIndexType>();
    }

    auto docCount = binarizedFeatures.size() / model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount();
    TVector<TIndexType> indexesVec(docCount);
    const auto* treeSplitsCurPtr =
        model.ObliviousTrees.GetRepackedBins().data() +
        model.ObliviousTrees.TreeStartOffsets[treeId];
    CalcIndexes(!model.ObliviousTrees.OneHotFeatures.empty(), binarizedFeatures.data(), docCount, indexesVec.data(), treeSplitsCurPtr, model.ObliviousTrees.TreeSizes[treeId]);
    return indexesVec;
}
