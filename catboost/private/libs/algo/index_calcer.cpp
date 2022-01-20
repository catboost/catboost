#include "index_calcer.h"

#include "features_data_helpers.h"
#include "fold.h"
#include "nonsymmetric_index_calcer.h"
#include "online_ctr.h"
#include "scoring.h"
#include "split.h"
#include "tree_print.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data/model_dataset_compatibility.h>
#include <catboost/libs/helpers/dense_hash.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/map_merge.h>
#include <catboost/libs/model/cpu/evaluator.h>
#include <catboost/libs/model/model.h>

#include <library/cpp/containers/stack_vector/stack_vec.h>
#include <library/cpp/threading/local_executor/local_executor.h>

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


template <typename TCount, typename TCmpOp, int VectorWidth>
inline void UpdateIndicesKernel(
    const TCount* histogram,
    TCmpOp cmpOp,
    int level,
    TIndexType* indices) {

    Y_ASSERT(VectorWidth == 4);
    const TCount hist0 = histogram[0];
    const TCount hist1 = histogram[1];
    const TCount hist2 = histogram[2];
    const TCount hist3 = histogram[3];
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
    const TCount* histogram,
    TIndexRange<ui32> indexRange,
    TCmpOp cmpOp,
    int level,
    TIndexType* indices) {

    constexpr int vectorWidth = 4;

    ui32 doc;
    for (doc = indexRange.Begin; doc + vectorWidth <= indexRange.End; doc += vectorWidth) {
        UpdateIndicesKernel<TCount, TCmpOp, vectorWidth>(
            histogram + doc,
            cmpOp,
            level,
            indices + doc);
    }
    for (; doc < indexRange.End; ++doc) {
        indices[doc] += cmpOp(histogram[doc]) * level;
    }
}

template <typename TColumn, class TCmpOp>
inline void ScheduleUpdateIndicesForSplit(
    const ui32* columnIndexingPtr, // can be nullptr
    const TColumn& column,
    TCmpOp cmpOp,
    int level,
    TIndexType* indices,
    TVector<std::function<void(TIndexRange<ui32>)>>* updateBlockCallbacks) {
    if (const auto* columnData
            = dynamic_cast<const TCompressedValuesHolderImpl<TColumn>*>(&column))
    {
        const TCompressedArray* compressedArray = columnData->GetCompressedData().GetSrc();

        updateBlockCallbacks->push_back(
            [columnIndexingPtr,
             cmpOp,
             level,
             indices,
             compressedArray]
                (TIndexRange<ui32> indexRange) {

                compressedArray->DispatchBitsPerKeyToDataType(
                    "UpdateIndicesForSplit",
                    [=] (const auto* histogram) {
                        if (columnIndexingPtr) {
                            UpdateIndicesForSplit(
                                columnIndexingPtr,
                                histogram,
                                indexRange,
                                cmpOp,
                                level,
                                indices);
                        } else {
                            UpdateIndicesForSplit(
                                histogram,
                                indexRange,
                                cmpOp,
                                level,
                                indices);
                        }
                    });
            });
    } else {
        CB_ENSURE_INTERNAL(false, "UpdateIndicesForSplit: unsupported column type");
    }
}


template <typename TColumn, class TCmpOp>
void ScheduleUpdateIndicesForSplit(
    TMaybe<TExclusiveBundleIndex> maybeExclusiveBundleIndex,
    TMaybe<TPackedBinaryIndex> maybeBinaryIndex,
    TMaybe<TFeaturesGroupIndex> maybeFeaturesGroupIndex,
    TConstArrayRef<TExclusiveFeaturesBundle> exclusiveFeaturesBundlesMetaData,
    const ui32* columnIndexing,  // can be nullptr
    const TColumn& column,
    std::function<const IExclusiveFeatureBundleArray*(ui32)>&& getExclusiveFeaturesBundle,
    std::function<const IBinaryPacksArray*(ui32)>&& getBinaryFeaturesPack,
    std::function<const IFeaturesGroupArray*(ui32)>&& getFeaturesGroup,
    TCmpOp cmpOp,
    int level,
    TIndexType* indices,
    TVector<std::function<void(TIndexRange<ui32>)>>* updateBlockCallbacks) {

    auto scheduleUpdateIndicesForSplit = [&] (const auto& column, auto&& cmpOp) {
        ScheduleUpdateIndicesForSplit(
            columnIndexing,
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
    const TOnlineCtrBase* OnlineCtr;
};


void GetObjectsDataAndIndexing(
    const TTrainingDataProviders& trainingData,
    const TFold& fold,
    bool isEstimated,
    bool isOnline,
    ui32 objectSubsetIdx, // 0 - learn, 1+ - test (subtract 1 for testIndex)
    TIndexedSubsetCache* indexedSubsetCache,
    NPar::ILocalExecutor* localExecutor,
    TQuantizedObjectsDataProviderPtr* objectsData,
    const ui32** columnIndexing // can return nullptr
) {
    if (isEstimated) {
        const auto& estimatedFeatures
            = isOnline ? fold.GetOnlineEstimatedFeatures() : trainingData.EstimatedObjectsData;
        *objectsData = objectSubsetIdx ? estimatedFeatures.Test[objectSubsetIdx - 1] : estimatedFeatures.Learn;
    } else {
        *objectsData = objectSubsetIdx ?
            trainingData.Test[objectSubsetIdx - 1]->ObjectsData
            : trainingData.Learn->ObjectsData;
    }
    if (isOnline) {
        *columnIndexing = nullptr;
    } else if (!objectSubsetIdx) {
        // learn
        if (isEstimated) {
            *columnIndexing = fold.GetLearnPermutationOfflineEstimatedFeaturesSubset().data();
        } else {
            *columnIndexing = fold.LearnPermutationFeaturesSubset.Get<TIndexedSubset<ui32>>().data();
        }
    } else {
        // test

        const TFeaturesArraySubsetIndexing& subsetIndexing = (*objectsData)->GetFeaturesArraySubsetIndexing();
        if (std::get_if<TFullSubset<ui32>>(&subsetIndexing)) {
            *columnIndexing = nullptr;
        } else if (const TIndexedSubset<ui32>* indexedSubset = std::get_if<TIndexedSubset<ui32>>(&subsetIndexing)) {
            *columnIndexing = indexedSubset->data();
        } else {
            // blocks

            TIndexedSubsetCache::insert_ctx insertCtx;
            auto it = indexedSubsetCache->find(&subsetIndexing, insertCtx);
            if (it != indexedSubsetCache->end()) {
                *columnIndexing = it->second.data();
            } else {
                TIndexedSubset<ui32> columnsIndexingStorage;
                columnsIndexingStorage.yresize(subsetIndexing.Size());
                subsetIndexing.ParallelForEach(
                    [&](ui32 idx, ui32 srcIdx) { columnsIndexingStorage[idx] = srcIdx; },
                    localExecutor);
                *columnIndexing = columnsIndexingStorage.data();
                indexedSubsetCache->emplace_direct(
                    insertCtx,
                    &subsetIndexing,
                    std::move(columnsIndexingStorage));
            }
        }

    }
}


static void UpdateIndices(
    bool initIndices,
    TConstArrayRef<TUpdateIndicesForSplitParams> params,
    const TTrainingDataProviders& trainingData,
    const TFold& fold,
    ui32 objectSubsetIdx, // 0 - learn, 1+ - test (subtract 1 for testIndex)
    NPar::ILocalExecutor* localExecutor,
    TArrayRef<TIndexType> indices) {

    if (indices.empty()) {
        return;
    }

    TIndexType defaultIndexValue = 0;

    TVector<std::function<void(TIndexRange<ui32>)>> updateBlockCallbacks;
    TIndexedSubsetCache indexedSubsetCache;

    TIndexType* indicesData = indices.data();

    for (const auto& splitParams : params) {
        const ui32 splitWeight = 1 << splitParams.Depth;
        const auto& split = splitParams.Split;

        if (split.Type == ESplitType::OnlineCtr) {
            const auto binBorder = split.BinBorder;
            const ui8* histogram = splitParams.OnlineCtr->GetData(split.Ctr, objectSubsetIdx).data();

            updateBlockCallbacks.push_back(
                [=] (TIndexRange<ui32> indexRange) {
                    UpdateIndicesForSplit(
                        histogram,
                        indexRange,
                        [=] (ui8 bucket) {
                            return IsTrueHistogram<ui8>(bucket, binBorder);
                        },
                        splitWeight,
                        indicesData);
                 }
            );
        } else {
            TQuantizedObjectsDataProviderPtr objectsDataProvider;
            const ui32* columnIndexing;
            GetObjectsDataAndIndexing(
                trainingData,
                fold,
                split.Type == ESplitType::EstimatedFeature,
                split.IsOnline(),
                objectSubsetIdx,
                &indexedSubsetCache,
                localExecutor,
                &objectsDataProvider,
                &columnIndexing);

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
                    objectsDataProvider->GetExclusiveFeatureBundlesMetaData(),
                    columnIndexing,
                    column,
                    [&] (ui32 bundleIdx) {
                        return &objectsDataProvider->GetExclusiveFeaturesBundle(bundleIdx);
                    },
                    [&] (ui32 packIdx) { return &objectsDataProvider->GetBinaryFeaturesPack(packIdx); },
                    [&] (ui32 groupIdx) {
                        return &objectsDataProvider->GetFeaturesGroup(groupIdx);
                    },
                    std::move(cmpOp),
                    splitWeight,
                    indicesData,
                    &updateBlockCallbacks);
            };


            if ((split.Type == ESplitType::FloatFeature) ||
                (split.Type == ESplitType::EstimatedFeature))
            {
                auto floatFeatureIdx = TFloatFeatureIdx((ui32)split.FeatureIdx);

                scheduleUpdateIndicesForSplit(
                    objectsDataProvider->GetFloatFeatureToExclusiveBundleIndex(floatFeatureIdx),
                    objectsDataProvider->GetFloatFeatureToPackedBinaryIndex(floatFeatureIdx),
                    objectsDataProvider->GetFloatFeatureToFeaturesGroupIndex(floatFeatureIdx),
                    **objectsDataProvider->GetFloatFeature((ui32)split.FeatureIdx),
                    [splitIdx = GetFeatureSplitIdx(split)] (ui16 bucket) {
                        return IsTrueHistogram<ui16>(bucket, splitIdx);
                    });
            } else {
                Y_ASSERT(split.Type == ESplitType::OneHotFeature);

                auto catFeatureIdx = TCatFeatureIdx((ui32)split.FeatureIdx);

                scheduleUpdateIndicesForSplit(
                    objectsDataProvider->GetCatFeatureToExclusiveBundleIndex(catFeatureIdx),
                    objectsDataProvider->GetCatFeatureToPackedBinaryIndex(catFeatureIdx),
                    objectsDataProvider->GetCatFeatureToFeaturesGroupIndex(catFeatureIdx),
                    **objectsDataProvider->GetCatFeature((ui32)split.FeatureIdx),
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
    const TTrainingDataProviders& trainingData,
    int curDepth,
    const TFold& fold,
    TArrayRef<TIndexType> indices,
    NPar::ILocalExecutor* localExecutor) {

    CB_ENSURE(curDepth > 0);

    const TOnlineCtrBase* onlineCtr = nullptr;
    if (split.Type == ESplitType::OnlineCtr) {
        onlineCtr = &fold.GetCtrs(split.Ctr.Projection);
    }

    TUpdateIndicesForSplitParams params{ (ui32)(curDepth - 1), split, onlineCtr };

    UpdateIndices(
        /*initIndices*/ false,
        TConstArrayRef<TUpdateIndicesForSplitParams>(&params, 1),
        trainingData,
        fold,
        0, // learn
        localExecutor,
        indices);
}

static TVector<bool> GetIsLeafEmptyOpt(ui64 leafCount, TConstArrayRef<TIndexType> indices, NPar::ILocalExecutor* localExecutor) {
    Y_ASSERT(leafCount <= 64);
    ui64 isLeafEmptyBits;
    MapMerge(
        localExecutor,
        TEqualRangesGenerator(TIndexRange(indices.ysize()), /*blockCount*/localExecutor->GetThreadCount() + 1),
        /*map*/[=] (const auto& range, ui64* output) {
            *output = leafCount == 64 ? -1 : (ui64(1) << leafCount) - 1;
            for (auto idx : range.Iter()) {
                const auto leafIdx = indices[idx];
                *output &= ~ (ui64(1) << ui64(leafIdx));
                if (*output == 0) {
                    break;
                }
            }
        },
        /*merge*/[] (ui64* isLeafEmptyBits, TVector<ui64>&& outputs) {
            for (auto output : outputs) {
                *isLeafEmptyBits &= output;
            }
        },
        &isLeafEmptyBits
    );
    TVector<bool> isLeafEmpty;
    isLeafEmpty.yresize(leafCount);
    for (auto idx : xrange(leafCount)) {
        isLeafEmpty[idx] = isLeafEmptyBits & ui64(1);
        isLeafEmptyBits = isLeafEmptyBits >> ui64(1);
    }
    return isLeafEmpty;
}

TVector<bool> GetIsLeafEmpty(int curDepth, TConstArrayRef<TIndexType> indices, NPar::ILocalExecutor* localExecutor) {
    const ui64 leafCount = ui64(1) << ui64(curDepth);
    if (leafCount <= 64) {
        return GetIsLeafEmptyOpt(leafCount, indices, localExecutor);
    }
    TVector<bool> isLeafEmpty;
    MapMerge(
        localExecutor,
        TEqualRangesGenerator(TIndexRange(indices.ysize()), /*blockCount*/localExecutor->GetThreadCount() + 1),
        /*map*/[=] (const auto& range, TVector<bool>* output) {
            output->resize(leafCount, true);
            size_t populatedLeafCount = 0;
            for (auto idx : range.Iter()) {
                const auto leafIdx = indices[idx];
                populatedLeafCount += (*output)[leafIdx];
                (*output)[leafIdx] = false;
                if (populatedLeafCount == leafCount) {
                    break;
                }
            }
        },
        /*merge*/[=] (TVector<bool>* isLeafEmpty, TVector<TVector<bool>>&& outputs) {
            for (const auto& output : outputs) {
                for (auto idx : xrange(leafCount)) {
                    (*isLeafEmpty)[idx] &= output[idx];
                }
            }
        },
        &isLeafEmpty);
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

// Get OnlineCtrBases associated with a fold
static TVector<const TOnlineCtrBase*> GetOnlineCtrs(const TFold& fold, const TSplitTree& tree) {
    TVector<const TOnlineCtrBase*> onlineCtrs(tree.GetDepth());
    for (int splitIdx = 0; splitIdx < tree.GetDepth(); ++splitIdx) {
        const auto& split = tree.Splits[splitIdx];
        if (split.Type == ESplitType::OnlineCtr) {
            onlineCtrs[splitIdx] = &fold.GetCtrs(split.Ctr.Projection);
        }
    }
    return onlineCtrs;
}

static TVector<const TOnlineCtrBase*> GetOnlineCtrs(
    const TFold& fold,
    const TNonSymmetricTreeStructure& tree) {

    const auto nodes = tree.GetNodes();
    TVector<const TOnlineCtrBase*> onlineCtrs(nodes.size());
    for (auto nodeIdx : xrange(nodes.size())) {
        const auto& split = nodes[nodeIdx].Split;
        if (split.Type == ESplitType::OnlineCtr) {
            onlineCtrs[nodeIdx] = &fold.GetCtrs(split.Ctr.Projection);
        }
    }
    return onlineCtrs;
}

static TVector<const TOnlineCtrBase*> GetOnlineCtrs(
    const TFold& fold,
    const std::variant<TSplitTree, TNonSymmetricTreeStructure>& tree) {

    if (std::holds_alternative<TSplitTree>(tree)) {
        return GetOnlineCtrs(fold, std::get<TSplitTree>(tree));
    } else {
        return GetOnlineCtrs(fold, std::get<TNonSymmetricTreeStructure>(tree));
    }
}

static void BuildIndicesForDataset(
    const TSplitTree& tree,
    const TTrainingDataProviders& trainingData,
    const TFold& fold,
    ui32 sampleCount,
    const TVector<const TOnlineCtrBase*>& onlineCtrs,
    ui32 objectSubsetIdx, // 0 - learn, 1+ - test (subtract 1 for testIndex)
    NPar::ILocalExecutor* localExecutor,
    TIndexType* indices) {

    TVector<TUpdateIndicesForSplitParams> params;
    params.reserve(tree.GetDepth());

    for (auto splitIdx : xrange(tree.GetDepth())) {
        params.push_back({(ui32)splitIdx, tree.Splits[splitIdx], onlineCtrs[splitIdx]});
    }

    UpdateIndices(
        /*initIndices*/ true,
        params,
        trainingData,
        fold,
        objectSubsetIdx,
        localExecutor,
        MakeArrayRef(indices, sampleCount));
}

static void BuildIndicesForDataset(
    const std::variant<TSplitTree, TNonSymmetricTreeStructure>& treeVariant,
    const TTrainingDataProviders& trainingData,
    const TFold& fold,
    ui32 sampleCount,
    const TVector<const TOnlineCtrBase*>& onlineCtrs,
    ui32 objectSubsetIdx, // 0 - learn, 1+ - test (subtract 1 for testIndex)
    NPar::ILocalExecutor* localExecutor,
    TIndexType* indices) {

    const auto buildIndices = [&](auto tree) {
        BuildIndicesForDataset(
            tree,
            trainingData,
            fold,
            sampleCount,
            onlineCtrs,
            objectSubsetIdx,
            localExecutor,
            indices);
    };

    if (std::holds_alternative<TSplitTree>(treeVariant)) {
        buildIndices(std::get<TSplitTree>(treeVariant));
    } else {
        buildIndices(std::get<TNonSymmetricTreeStructure>(treeVariant));
    }
}


TVector<TIndexType> BuildIndices(
    const TFold& fold,
    const std::variant<TSplitTree, TNonSymmetricTreeStructure>& tree,
    const TTrainingDataProviders& trainingData,
    EBuildIndicesDataParts dataParts,
    NPar::ILocalExecutor* localExecutor) {

    ui32 learnSampleCount
        = (dataParts == EBuildIndicesDataParts::TestOnly) ? 0 : trainingData.Learn->GetObjectCount();
    ui32 tailSampleCount
        = (dataParts == EBuildIndicesDataParts::LearnOnly) ? 0 : trainingData.GetTestSampleCount();

    const TVector<const TOnlineCtrBase*>& onlineCtrs = GetOnlineCtrs(fold, tree);

    TVector<TIndexType> indices;
    indices.yresize(learnSampleCount + tailSampleCount);

    if (dataParts != EBuildIndicesDataParts::TestOnly) {
        BuildIndicesForDataset(
            tree,
            trainingData,
            fold,
            learnSampleCount,
            onlineCtrs,
            /*objectSubsetIdx*/ 0, // learn
            localExecutor,
            indices.begin());
    }
    if (dataParts != EBuildIndicesDataParts::LearnOnly) {
        ui32 docOffset = learnSampleCount;
        for (size_t testIdx = 0; testIdx < trainingData.Test.size(); ++testIdx) {
            const auto& testSet = *trainingData.Test[testIdx];
            BuildIndicesForDataset(
                tree,
                trainingData,
                fold,
                testSet.GetObjectCount(),
                onlineCtrs,
                testIdx + 1,
                localExecutor,
                indices.begin() + docOffset);
            docOffset += testSet.GetObjectCount();
        }
    }
    return indices;
}

TVector<TIndexType> BuildIndicesForBinTree(
    const TFullModel& model,
    const NCB::NModelEvaluation::IQuantizedData* quantizedFeatures,
    size_t treeId) {

    if (model.ModelTrees->GetEffectiveBinaryFeaturesBucketsCount() == 0) {
        return TVector<TIndexType>();
    }
    TVector<TIndexType> indexesVec(quantizedFeatures->GetObjectsCount());
    auto evaluator =  model.GetCurrentEvaluator();
    evaluator->CalcLeafIndexes(quantizedFeatures, treeId, treeId + 1, indexesVec);
    return indexesVec;
}

