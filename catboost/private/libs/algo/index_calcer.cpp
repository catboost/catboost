#include "index_calcer.h"

#include "features_data_helpers.h"
#include "fold.h"
#include "online_ctr.h"
#include "scoring.h"
#include "split.h"
#include "tree_print.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data/model_dataset_compatibility.h>
#include <catboost/libs/helpers/dense_hash.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/cpu/evaluator.h>
#include <catboost/libs/model/model.h>

#include <library/containers/stack_vector/stack_vec.h>
#include <library/threading/local_executor/local_executor.h>

#include <functional>


using namespace NCB;


static inline ui16 GetFeatureSplitIdx(const TSplit& split) {
    return split.BinBorder;
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
    const ui32* permutation,
    const TCount* histogram,
    TIndexRange<ui32> indexRange,
    TCmpOp cmpOp,
    int level,
    TIndexType* indices) {

    constexpr int vectorWidth = 4;

    ui32 doc;
    for (doc = indexRange.Begin; doc + vectorWidth <= indexRange.End; doc += vectorWidth) {
        UpdateIndicesKernel<TCount, TCmpOp, vectorWidth>(
            permutation + doc,
            histogram,
            cmpOp,
            level,
            indices + doc);
    }
    for (; doc < indexRange.End; ++doc) {
        const int idxOriginal = permutation[doc];
        indices[doc] += cmpOp(histogram[idxOriginal]) * level;
    }
}

template <typename T, EFeatureValuesType FeatureValuesType, class TCmpOp>
inline void ScheduleUpdateIndicesForSplit(
    const TIndexedSubset<ui32>& columnsIndexing,
    const TTypedFeatureValuesHolder<T, FeatureValuesType>& column,
    TCmpOp cmpOp,
    int level,
    TIndexType* indices,
    TVector<std::function<void(TIndexRange<ui32>)>>* updateBlockCallbacks) {

    if (const auto* columnData
            = dynamic_cast<const TCompressedValuesHolderImpl<T, FeatureValuesType>*>(&column))
    {
        const auto* columnsIndexingPtr = &columnsIndexing;
        const TCompressedArray* compressedArray = columnData->GetCompressedData().GetSrc();

        updateBlockCallbacks->push_back(
            [columnsIndexingPtr,
             cmpOp,
             level,
             indices,
             compressedArray]
                (TIndexRange<ui32> indexRange) {

                NCB::DispatchBitsPerKeyToDataType(
                    *compressedArray,
                    "UpdateIndicesForSplit",
                    [=] (const auto* histogram) {
                        UpdateIndicesForSplit(
                            columnsIndexingPtr->data(),
                            histogram,
                            indexRange,
                            cmpOp,
                            level,
                            indices);
                    });
            });
    } else {
        CB_ENSURE_INTERNAL(false, "UpdateIndicesForSplit: unsupported column type");
    }
}


template <typename T, EFeatureValuesType FeatureValuesType, class TCmpOp>
void ScheduleUpdateIndicesForSplit(
    TMaybe<TExclusiveBundleIndex> maybeExclusiveBundleIndex,
    TMaybe<TPackedBinaryIndex> maybeBinaryIndex,
    TMaybe<TFeaturesGroupIndex> maybeFeaturesGroupIndex,
    TConstArrayRef<TExclusiveFeaturesBundle> exclusiveFeaturesBundlesMetaData,
    const TIndexedSubset<ui32>& columnsIndexing,
    const TTypedFeatureValuesHolder<T, FeatureValuesType>& column,
    std::function<const TExclusiveFeatureBundleHolder*(ui32)>&& getExclusiveFeaturesBundle,
    std::function<const TBinaryPacksHolder*(ui32)>&& getBinaryFeaturesPack,
    std::function<const TFeaturesGroupHolder*(ui32)>&& getFeaturesGroup,
    TCmpOp cmpOp,
    int level,
    TIndexType* indices,
    TVector<std::function<void(TIndexRange<ui32>)>>* updateBlockCallbacks) {

    auto scheduleUpdateIndicesForSplit = [&] (const auto& column, auto&& cmpOp) {
        ScheduleUpdateIndicesForSplit(
            columnsIndexing,
            column,
            std::move(cmpOp),
            level,
            indices,
            updateBlockCallbacks);
    };

    if (maybeBinaryIndex) {
        TBinaryFeaturesPack bitMask = TBinaryFeaturesPack(1) << maybeBinaryIndex->BitIdx;
        TBinaryFeaturesPack bitIdx = maybeBinaryIndex->BitIdx;

        scheduleUpdateIndicesForSplit(
            *getBinaryFeaturesPack(maybeBinaryIndex->PackIdx),
            [bitMask, bitIdx, cmpOp = std::move(cmpOp)] (NCB::TBinaryFeaturesPack featuresPack) {
                return cmpOp((featuresPack & bitMask) >> bitIdx);
            });
    } else if (maybeExclusiveBundleIndex) {
        auto boundsInBundle
            = exclusiveFeaturesBundlesMetaData[maybeExclusiveBundleIndex->BundleIdx]
                .Parts[maybeExclusiveBundleIndex->InBundleIdx].Bounds;

        scheduleUpdateIndicesForSplit(
            *getExclusiveFeaturesBundle(maybeExclusiveBundleIndex->BundleIdx),
            [boundsInBundle, cmpOp = std::move(cmpOp)] (ui16 featuresBundle) {
                return cmpOp(GetBinFromBundle<ui16>(featuresBundle, boundsInBundle));
            });
    } else if (maybeFeaturesGroupIndex) {
        scheduleUpdateIndicesForSplit(
            *getFeaturesGroup(maybeFeaturesGroupIndex->GroupIdx),
            [partIdx = maybeFeaturesGroupIndex->InGroupIdx, cmpOp = std::move(cmpOp)] (const auto& featuresGroupValue) {
                return cmpOp(GetPartValueFromGroup(featuresGroupValue, partIdx));
            });
    } else {
        scheduleUpdateIndicesForSplit(column, std::move(cmpOp));
    }
}

struct TUpdateIndicesForSplitParams {
    ui32 Depth;
    const TSplit& Split;
    const TOnlineCTR* OnlineCtr;
};


static TConstArrayRef<ui8> GetCtrValues(const TSplit& split, const TOnlineCTR& ctr) {
    return ctr.Feature[split.Ctr.CtrIdx][split.Ctr.TargetBorderIdx][split.Ctr.PriorIdx];
}


static void UpdateIndices(
    bool initIndices,
    TConstArrayRef<TUpdateIndicesForSplitParams> params,
    ui32 onlineCtrObjectOffset,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const TIndexedSubset<ui32>& columnsIndexing,
    NPar::TLocalExecutor* localExecutor,
    TArrayRef<TIndexType> indices) {

    TIndexType defaultIndexValue = 0;

    TVector<std::function<void(TIndexRange<ui32>)>> updateBlockCallbacks;

    TIndexType* indicesData = indices.data();

    for (const auto& splitParams : params) {
        const ui32 splitWeight = 1 << splitParams.Depth;
        const auto& split = splitParams.Split;

        if (split.Type == ESplitType::OnlineCtr) {
            const auto ctr = splitParams.OnlineCtr;
            const auto ctrValuesData = GetCtrValues(split, *ctr).data() + onlineCtrObjectOffset;
            const auto binBorder = split.BinBorder;
            updateBlockCallbacks.push_back(
                [=] (TIndexRange<ui32> indexRange) {
                    const ui32 begin = *indexRange.Iter().begin();
                    const ui32 end = *indexRange.Iter().end();
                    for (ui32 i = begin; i < end; ++i) {
                        indicesData[i] += (ctrValuesData[i] > binBorder) * splitWeight;
                    }
                });
        } else {
            auto scheduleUpdateIndicesForSplit = [&] (
                auto maybeExclusiveBundleIndex,
                auto maybeBinaryIndex,
                auto maybeFeaturesGroupIndex,
                const auto& column,
                auto&& cmpOp) {

                ScheduleUpdateIndicesForSplit(
                    maybeExclusiveBundleIndex,
                    maybeBinaryIndex,
                    maybeFeaturesGroupIndex,
                    objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
                    columnsIndexing,
                    column,
                    [&] (ui32 bundleIdx) {
                        return &objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx);
                    },
                    [&] (ui32 packIdx) { return &objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                    [&] (ui32 groupIdx) {
                        return &objectsDataProvider.GetFeaturesGroup(groupIdx);
                    },
                    std::move(cmpOp),
                    splitWeight,
                    indicesData,
                    &updateBlockCallbacks);
            };


            if (split.Type == ESplitType::FloatFeature) {
                auto floatFeatureIdx = TFloatFeatureIdx((ui32)split.FeatureIdx);

                scheduleUpdateIndicesForSplit(
                    objectsDataProvider.GetFloatFeatureToExclusiveBundleIndex(floatFeatureIdx),
                    objectsDataProvider.GetFloatFeatureToPackedBinaryIndex(floatFeatureIdx),
                    objectsDataProvider.GetFloatFeatureToFeaturesGroupIndex(floatFeatureIdx),
                    **objectsDataProvider.GetFloatFeature((ui32)split.FeatureIdx),
                    [splitIdx = GetFeatureSplitIdx(split)] (ui16 bucket) {
                        return IsTrueHistogram<ui16>(bucket, splitIdx);
                    });
            } else {
                Y_ASSERT(split.Type == ESplitType::OneHotFeature);

                auto catFeatureIdx = TCatFeatureIdx((ui32)split.FeatureIdx);

                scheduleUpdateIndicesForSplit(
                    objectsDataProvider.GetCatFeatureToExclusiveBundleIndex(catFeatureIdx),
                    objectsDataProvider.GetCatFeatureToPackedBinaryIndex(catFeatureIdx),
                    objectsDataProvider.GetCatFeatureToFeaturesGroupIndex(catFeatureIdx),
                    **objectsDataProvider.GetCatFeature((ui32)split.FeatureIdx),
                    [bucketIdx = (ui32)split.BinBorder] (ui32 bucket) {
                        return IsTrueOneHotFeature(bucket, bucketIdx);
                    });
            }
        }
    }

    const ui32 blockSize = 1000;

    TSimpleIndexRangesGenerator<ui32> indexRanges(
        TIndexRange<ui32>(SafeIntegerCast<ui32>(indices.size())),
        blockSize);

    NPar::ParallelFor(
        *localExecutor,
        0,
        SafeIntegerCast<int>(indexRanges.RangesCount()),
        [&, defaultIndexValue] (int blockIdx) {
            auto indexRange = indexRanges.GetRange((int)blockIdx);

            if (initIndices) {
                for (auto i : indexRange.Iter()) {
                    indicesData[i] = defaultIndexValue;
                }
            } else {
                for (auto i : indexRange.Iter()) {
                    indicesData[i] += defaultIndexValue;
                }
            }

            for (auto& updateBlockCallback : updateBlockCallbacks) {
                updateBlockCallback(indexRange);
            }
        });
}

void SetPermutedIndices(
    const TSplit& split,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    int curDepth,
    const TFold& fold,
    TVector<TIndexType>* indices,
    NPar::TLocalExecutor* localExecutor) {

    CB_ENSURE(curDepth > 0);

    const TOnlineCTR* onlineCtr = nullptr;
    if (split.Type == ESplitType::OnlineCtr) {
        onlineCtr = &fold.GetCtr(split.Ctr.Projection);
    }

    TUpdateIndicesForSplitParams params{ (ui32)(curDepth - 1), split, onlineCtr };

    UpdateIndices(
        /*initIndices*/ false,
        TConstArrayRef<TUpdateIndicesForSplitParams>(&params, 1),
        /*onlineCtrObjectOffset*/ 0,
        objectsDataProvider,
        fold.LearnPermutationFeaturesSubset.Get<TIndexedSubset<ui32>>(),
        localExecutor,
        *indices);
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
    ui32 docOffset,
    NPar::TLocalExecutor* localExecutor,
    TIndexType* indices) {

    TIndexedSubset<ui32> columnsIndexingStorage;

    const TIndexedSubset<ui32>* columnsIndexing
        = GetIf<TIndexedSubset<ui32>>(&featuresArraySubsetIndexing);

    if (!columnsIndexing) {
        columnsIndexingStorage.yresize(featuresArraySubsetIndexing.Size());
        featuresArraySubsetIndexing.ParallelForEach(
            [&](ui32 idx, ui32 srcIdx) { columnsIndexingStorage[idx] = srcIdx; },
            localExecutor);
        columnsIndexing = &columnsIndexingStorage;
    }


    TVector<TUpdateIndicesForSplitParams> params;
    params.reserve(tree.GetDepth());

    for (auto splitIdx : xrange(tree.GetDepth())) {
        params.push_back({(ui32)splitIdx, tree.Splits[splitIdx], onlineCtrs[splitIdx]});
    }

    UpdateIndices(
        /*initIndices*/ true,
        params,
        docOffset,
        objectsDataProvider,
        *columnsIndexing,
        localExecutor,
        MakeArrayRef(indices, sampleCount));
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

    TVector<TIndexType> indices;
    indices.yresize(learnSampleCount + tailSampleCount);

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
            docOffset,
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

