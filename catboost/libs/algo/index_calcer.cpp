#include "index_calcer.h"

#include "features_data_helpers.h"
#include "fold.h"
#include "online_ctr.h"
#include "score_calcer.h"
#include "split.h"
#include "tree_print.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data_new/model_dataset_compatibility.h>
#include <catboost/libs/helpers/dense_hash.h>
#include <catboost/libs/model/cpu/evaluator.h>
#include <catboost/libs/model/model.h>

#include <library/containers/stack_vector/stack_vec.h>
#include <library/threading/local_executor/local_executor.h>

#include <functional>


using namespace NCB;


static bool GetCtrSplit(const TSplit& split, int idxPermuted, const TOnlineCTR& ctr) {
    ui8 ctrValue = ctr.Feature[split.Ctr.CtrIdx]
                              [split.Ctr.TargetBorderIdx]
                              [split.Ctr.PriorIdx]
                              [idxPermuted];
    return ctrValue > split.BinBorder;
}

static inline ui16 GetFeatureSplitIdx(const TSplit& split) {
    return split.BinBorder;
}

static inline const TVariant<const ui8*, const ui16*> GetFloatHistogram(
    const TSplit& split,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider) {

    const auto* featureColumnHolder = *objectsDataProvider.GetNonPackedFloatFeature((ui32)split.FeatureIdx);
    if (featureColumnHolder->GetBitsPerKey() == 8) {
        return *featureColumnHolder->GetArrayData<ui8>().GetSrc();
    } else {
        return *featureColumnHolder->GetArrayData<ui16>().GetSrc();
    }
}

static inline const ui32* GetRemappedCatFeatures(
    const TSplit& split,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider) {

    return *(*objectsDataProvider.GetNonPackedCatFeature((ui32)split.FeatureIdx))
        ->GetArrayData<ui32>().GetSrc();
}

template <typename TCount, typename TCmpOp, int VectorWidth>
inline void UpdateIndicesKernel(
    const ui32* permutation,
    const TCount* histogram,
    TCmpOp cmpOp,
    int level,
    TIndexType* indices) {

    Y_ASSERT(VectorWidth == 4);
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
inline void UpdateIndicesForSplit(
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
        UpdateIndicesKernel<TCount, TCmpOp, vectorWidth>(
            permutation + doc,
            histogram,
            cmpOp,
            level,
            indices + doc);
    }
    for (; doc < nextBlockStart; ++doc) {
        const int idxOriginal = permutation[doc];
        indices[doc] += cmpOp(histogram[idxOriginal]) * level;
    }
}


template <typename TCount, class TCmpOp>
inline void UpdateIndicesForSplit(
    const NPar::TLocalExecutor::TExecRangeParams& params,
    int blockIdx,
    TMaybe<TExclusiveBundleIndex> maybeExclusiveBundleIndex,
    TMaybe<TPackedBinaryIndex> maybeBinaryIndex,
    const ui32* permutation,
    const TCount* histogram, // can be nullptr if maybeBinaryIndex
    std::function<TFeaturesBundleArraySubset(ui32)>&& getExclusiveFeaturesBundle,
    std::function<TPackedBinaryFeaturesArraySubset(ui32)>&& getBinaryFeaturesPack,
    TCmpOp cmpOp,
    int level,
    TIndexType* indices) {

    if (maybeBinaryIndex) {
        TBinaryFeaturesPack bitMask = TBinaryFeaturesPack(1) << maybeBinaryIndex->BitIdx;
        TBinaryFeaturesPack bitIdx = maybeBinaryIndex->BitIdx;

        NCB::TPackedBinaryFeaturesArraySubset packSubset = getBinaryFeaturesPack(maybeBinaryIndex->PackIdx);

        UpdateIndicesForSplit(
            params,
            blockIdx,
            permutation,
            (**packSubset.GetSrc()).data(),
            [bitMask, bitIdx, cmpOp = std::move(cmpOp)] (NCB::TBinaryFeaturesPack featuresPack) {
                return cmpOp((featuresPack & bitMask) >> bitIdx);
            },
            level,
            indices);
    } else if (maybeExclusiveBundleIndex) {

        TFeaturesBundleArraySubset bundleSubset
            = getExclusiveFeaturesBundle(maybeExclusiveBundleIndex->BundleIdx);

        auto boundsInBundle = bundleSubset.MetaData->Parts[maybeExclusiveBundleIndex->InBundleIdx].Bounds;

        auto updateIndicesForSplit = [&] (const auto* histogram) {
            UpdateIndicesForSplit(
                params,
                blockIdx,
                permutation,
                histogram,
                [boundsInBundle, cmpOp = std::move(cmpOp)] (auto featuresBundle) {
                    return cmpOp(GetBinFromBundle<TCount>(featuresBundle, boundsInBundle));
                },
                level,
                indices);
        };

        switch (bundleSubset.MetaData->SizeInBytes) {
            case 1:
                updateIndicesForSplit(bundleSubset.SrcData.data());
                break;
            case 2:
                updateIndicesForSplit((const ui16*)bundleSubset.SrcData.data());
                break;
            default:
                CB_ENSURE_INTERNAL(
                    false,
                    "unsupported Bundle SizeInBytes = " << bundleSubset.MetaData->SizeInBytes);
        }
    } else {
        UpdateIndicesForSplit(
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

        TVariant<const ui8*, const ui16*> histogram;
        auto maybeExclusiveFeaturesBundleIndex
            = objectsDataProvider.GetFloatFeatureToExclusiveBundleIndex(floatFeatureIdx);
        auto maybeBinaryIndex = objectsDataProvider.GetFloatFeatureToPackedBinaryIndex(floatFeatureIdx);
        if (!maybeExclusiveFeaturesBundleIndex && !maybeBinaryIndex) {
            histogram = GetFloatHistogram(split, objectsDataProvider);
        }

        localExecutor->ExecRange(
            [&](int blockIdx) {
                if (HoldsAlternative<const ui8*>(histogram)) {
                    UpdateIndicesForSplit(
                        blockParams,
                        blockIdx,
                        maybeExclusiveFeaturesBundleIndex,
                        maybeBinaryIndex,
                        fold.LearnPermutationFeaturesSubset.Get<TIndexedSubset<ui32>>().data(),
                        Get<const ui8*>(histogram),
                        [&] (ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
                        [&] (ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                        [splitIdx = GetFeatureSplitIdx(split)] (ui8 bucket) {
                            return IsTrueHistogram<ui8>(bucket, splitIdx);
                        },
                        splitWeight,
                        indicesData);
                } else {
                    UpdateIndicesForSplit(
                        blockParams,
                        blockIdx,
                        maybeExclusiveFeaturesBundleIndex,
                        maybeBinaryIndex,
                        fold.LearnPermutationFeaturesSubset.Get<TIndexedSubset<ui32>>().data(),
                        Get<const ui16*>(histogram),
                        [&] (ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
                        [&] (ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                        [splitIdx = GetFeatureSplitIdx(split)] (ui16 bucket) {
                            return IsTrueHistogram<ui16>(bucket, splitIdx);
                        },
                        splitWeight,
                        indicesData);
                }
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
        auto maybeExclusiveFeaturesBundleIndex
            = objectsDataProvider.GetCatFeatureToExclusiveBundleIndex(catFeatureIdx);
        if (!maybeExclusiveFeaturesBundleIndex && !maybeBinaryIndex) {
            histogram = GetRemappedCatFeatures(split, objectsDataProvider);
        }

        localExecutor->ExecRange(
            [&] (int blockIdx) {
                UpdateIndicesForSplit(
                    blockParams,
                    blockIdx,
                    maybeExclusiveFeaturesBundleIndex,
                    maybeBinaryIndex,
                    fold.LearnPermutationFeaturesSubset.Get<TIndexedSubset<ui32>>().data(),
                    histogram,
                    [&] (ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
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
    size_t populatedLeafCount = 0;
    for (auto idx : indices) {
        populatedLeafCount += isLeafEmpty[idx];
        isLeafEmpty[idx] = false;
        if (populatedLeafCount == (1 << curDepth)) {
            break;
        }
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
            localExecutor);
        permutation = permutationStorage.data();
    }

    const int blockSize = 1000;
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, (int)sampleCount);
    blockParams.SetBlockSize(blockSize);

    // precalc to avoid recalculation in each block
    TVector<TVariant<const ui8*, const ui16*>> splitFloatHistograms;
    splitFloatHistograms.yresize(tree.GetDepth());

    TVector<const ui32*> splitRemappedCatHistograms;
    splitRemappedCatHistograms.yresize(tree.GetDepth());

    for (auto splitIdx : xrange(tree.GetDepth())) {
        const auto& split = tree.Splits[splitIdx];
        if (split.Type == ESplitType::FloatFeature) {
            auto floatFeatureIdx = TFloatFeatureIdx((ui32)split.FeatureIdx);
            if (!objectsDataProvider.IsFeaturePackedBinary(floatFeatureIdx) &&
                !objectsDataProvider.IsFeatureInExclusiveBundle(floatFeatureIdx))
            {
                splitFloatHistograms[splitIdx] = GetFloatHistogram(split, objectsDataProvider);
            }
        } else if (split.Type == ESplitType::OneHotFeature) {
            auto catFeatureIdx = TCatFeatureIdx((ui32)split.FeatureIdx);
            if (!objectsDataProvider.IsFeaturePackedBinary(catFeatureIdx) &&
                !objectsDataProvider.IsFeatureInExclusiveBundle(catFeatureIdx))
            {
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
                if (HoldsAlternative<const ui8*>(splitFloatHistograms[splitIdx])) {
                    UpdateIndicesForSplit(
                        blockParams,
                        blockIdx,
                        objectsDataProvider.GetFloatFeatureToExclusiveBundleIndex(floatFeatureIdx),
                        objectsDataProvider.GetFloatFeatureToPackedBinaryIndex(floatFeatureIdx),
                        permutation,
                        Get<const ui8*>(splitFloatHistograms[splitIdx]),
                        [&](ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
                        [&](ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                        [splitIdx = GetFeatureSplitIdx(split)](ui8 bucket) {
                            return IsTrueHistogram<ui8>(bucket, splitIdx);
                        },
                        splitWeight,
                        indices);
                } else {
                    UpdateIndicesForSplit(
                        blockParams,
                        blockIdx,
                        objectsDataProvider.GetFloatFeatureToExclusiveBundleIndex(floatFeatureIdx),
                        objectsDataProvider.GetFloatFeatureToPackedBinaryIndex(floatFeatureIdx),
                        permutation,
                        Get<const ui16*>(splitFloatHistograms[splitIdx]),
                        [&](ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
                        [&](ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                        [splitIdx = GetFeatureSplitIdx(split)](ui16 bucket) {
                            return IsTrueHistogram<ui16>(bucket, splitIdx);
                        },
                        splitWeight,
                        indices);
                }
            } else if (split.Type == ESplitType::OnlineCtr) {
                const TOnlineCTR& splitOnlineCtr = *onlineCtrs[splitIdx];
                NPar::TLocalExecutor::BlockedLoopBody(
                    blockParams,
                    [&](int doc) {
                        indices[doc] += GetCtrSplit(split, doc + docOffset, splitOnlineCtr) * splitWeight;
                    })(blockIdx);
            } else {
                Y_ASSERT(split.Type == ESplitType::OneHotFeature);

                auto catFeatureIdx = TCatFeatureIdx((ui32)split.FeatureIdx);

                UpdateIndicesForSplit(
                    blockParams,
                    blockIdx,
                    objectsDataProvider.GetCatFeatureToExclusiveBundleIndex(catFeatureIdx),
                    objectsDataProvider.GetCatFeatureToPackedBinaryIndex(catFeatureIdx),
                    permutation,
                    splitRemappedCatHistograms[splitIdx],
                    [&] (ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
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

TVector<TIndexType> BuildIndicesForBinTree(
    const TFullModel& model,
    const NCB::NModelEvaluation::IQuantizedData* quantizedFeatures,
    size_t treeId) {

    if (model.ObliviousTrees->GetEffectiveBinaryFeaturesBucketsCount() == 0) {
        return TVector<TIndexType>();
    }
    TVector<TIndexType> indexesVec(quantizedFeatures->GetObjectsCount());
    auto evaluator =  model.GetCurrentEvaluator();
    evaluator->CalcLeafIndexes(quantizedFeatures, treeId, treeId + 1, indexesVec);
    return indexesVec;
}

