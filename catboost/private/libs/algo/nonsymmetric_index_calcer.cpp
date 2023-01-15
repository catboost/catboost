#include "nonsymmetric_index_calcer.h"

#include "index_calcer.h"

#include "fold.h"
#include "online_ctr.h"
#include "split.h"

#include <catboost/libs/data/objects.h>
#include <catboost/libs/helpers/array_subset.h>

#include <util/generic/vector.h>

using namespace NCB;


template <typename TColumn, class TCmpOp>
inline std::function<bool(ui32)> BuildNodeSplitFunction(
    const TColumn& column,
    TCmpOp cmpOp) {

    if (const auto* columnData
        = dynamic_cast<const TCompressedValuesHolderImpl<TColumn>*>(&column))
    {
        const TCompressedArray* compressedArray = columnData->GetCompressedData().GetSrc();

        std::function<bool(ui32)> func;
        compressedArray->DispatchBitsPerKeyToDataType(
            "BuildNodeSplitFunction",
            [&func, cmpOp=std::move(cmpOp)] (const auto* featureData) {
                func = [featureData, cmpOp=std::move(cmpOp)] (ui32 objIdx) {
                    return cmpOp(featureData[objIdx]);
                };
            });
        return func;
    } else {
        CB_ENSURE_INTERNAL(false, "UpdateIndicesForSplit: unsupported column type");
    }
}

template <typename TColumn, class TCmpOp>
std::function<bool(ui32)> BuildNodeSplitFunction(
    TMaybe<TExclusiveBundleIndex> maybeExclusiveBundleIndex,
    TMaybe<TPackedBinaryIndex> maybeBinaryIndex,
    TMaybe<TFeaturesGroupIndex> maybeFeaturesGroupIndex,
    TConstArrayRef<TExclusiveFeaturesBundle> exclusiveFeaturesBundlesMetaData,
    const TColumn& column,
    std::function<const IExclusiveFeatureBundleArray*(ui32)>&& getExclusiveFeaturesBundle,
    std::function<const IBinaryPacksArray*(ui32)>&& getBinaryFeaturesPack,
    std::function<const IFeaturesGroupArray*(ui32)>&& getFeaturesGroup,
    TCmpOp cmpOp) {

    auto buildNodeSplitFunction = [&] (const auto& column, auto&& cmpOp) {
        return BuildNodeSplitFunction(
            column,
            std::move(cmpOp));
    };

    if (maybeBinaryIndex) {
        TBinaryFeaturesPack bitMask = TBinaryFeaturesPack(1) << maybeBinaryIndex->BitIdx;
        TBinaryFeaturesPack bitIdx = maybeBinaryIndex->BitIdx;

        return buildNodeSplitFunction(
            *getBinaryFeaturesPack(maybeBinaryIndex->PackIdx),
            [bitMask, bitIdx, cmpOp = std::move(cmpOp)] (NCB::TBinaryFeaturesPack featuresPack) {
                return cmpOp((featuresPack & bitMask) >> bitIdx);
            });
    } else if (maybeExclusiveBundleIndex) {
        auto boundsInBundle
            = exclusiveFeaturesBundlesMetaData[maybeExclusiveBundleIndex->BundleIdx]
                .Parts[maybeExclusiveBundleIndex->InBundleIdx].Bounds;

        return buildNodeSplitFunction(
            *getExclusiveFeaturesBundle(maybeExclusiveBundleIndex->BundleIdx),
            [boundsInBundle, cmpOp = std::move(cmpOp)] (ui16 featuresBundle) {
                return cmpOp(GetBinFromBundle<ui16>(featuresBundle, boundsInBundle));
            });
    } else if (maybeFeaturesGroupIndex) {
        return buildNodeSplitFunction(
            *getFeaturesGroup(maybeFeaturesGroupIndex->GroupIdx),
            [partIdx = maybeFeaturesGroupIndex->InGroupIdx, cmpOp = std::move(cmpOp)] (const auto& featuresGroupValue) {
                return cmpOp(GetPartValueFromGroup(featuresGroupValue, partIdx));
            });
    } else {
        return buildNodeSplitFunction(column, std::move(cmpOp));
    }
}

std::function<bool(ui32)> BuildNodeSplitFunction(
    const TSplitNode& node,
    const TQuantizedObjectsDataProvider& objectsDataProvider,
    const TOnlineCtrBase* onlineCtr,
    ui32 datasetIdx) {

    const auto& split = node.Split;

    if (split.Type == ESplitType::OnlineCtr) {
        const auto ctrValuesData = onlineCtr->GetData(split.Ctr, datasetIdx).data();
        const auto binBorder = split.BinBorder;
        return [ctrValuesData, binBorder](ui32 objIdx) {
            return ctrValuesData[objIdx] > binBorder;
        };
    } else {
        auto buildNodeSplitFunction = [&] (
            auto maybeExclusiveBundleIndex,
            auto maybeBinaryIndex,
            auto maybeFeaturesGroupIndex,
            const auto& column,
            auto&& cmpOp) {

            return BuildNodeSplitFunction(
                maybeExclusiveBundleIndex,
                maybeBinaryIndex,
                maybeFeaturesGroupIndex,
                objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
                column,
                [&] (ui32 bundleIdx) {
                    return &objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx);
                },
                [&] (ui32 packIdx) { return &objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                [&] (ui32 groupIdx) {
                    return &objectsDataProvider.GetFeaturesGroup(groupIdx);
                },
                std::move(cmpOp));
        };


        if ((split.Type == ESplitType::FloatFeature) || (split.Type == ESplitType::EstimatedFeature)){
            auto floatFeatureIdx = TFloatFeatureIdx((ui32)split.FeatureIdx);

            return buildNodeSplitFunction(
                objectsDataProvider.GetFloatFeatureToExclusiveBundleIndex(floatFeatureIdx),
                objectsDataProvider.GetFloatFeatureToPackedBinaryIndex(floatFeatureIdx),
                objectsDataProvider.GetFloatFeatureToFeaturesGroupIndex(floatFeatureIdx),
                **objectsDataProvider.GetFloatFeature((ui32)split.FeatureIdx),
                [binBorder = split.BinBorder] (ui16 bucket) {
                    return IsTrueHistogram<ui16>(bucket, binBorder);
                });
        } else {
            Y_ASSERT(split.Type == ESplitType::OneHotFeature);

            auto catFeatureIdx = TCatFeatureIdx((ui32)split.FeatureIdx);

            return buildNodeSplitFunction(
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

void UpdateIndices(
    const TSplitNode& node,
    const TTrainingDataProviders& trainingData,
    const TIndexedSubset<ui32>& docsSubset,
    const TFold& fold,
    NPar::ILocalExecutor* localExecutor,
    TArrayRef<TIndexType> indicesRef
) {
    TQuantizedObjectsDataProviderPtr objectsDataProvider;
    const ui32* columnsIndexing;
    TIndexedSubsetCache indexedSubsetCache; // not really updated because it is used for test only
    GetObjectsDataAndIndexing(
        trainingData,
        fold,
        node.Split.Type == ESplitType::EstimatedFeature,
        node.Split.IsOnline(),
        /*objectSubsetIdx*/ 0, // 0 - learn
        &indexedSubsetCache,
        localExecutor,
        &objectsDataProvider,
        &columnsIndexing // can return nullptr
    );

    auto func = BuildNodeSplitFunction(
        node,
        *objectsDataProvider,
        node.Split.Type == ESplitType::OnlineCtr ? &fold.GetCtrs(node.Split.Ctr.Projection) : nullptr,
        /*datasetIdx*/ 0);

    // TODO(ilyzhin) std::function is very slow for calling many times (maybe replace it with lambda)
    std::function<bool(ui32)> splitFunction;
    if (!columnsIndexing) {
        splitFunction = std::move(func);
    } else {
        splitFunction = [realObjIdx = columnsIndexing, func=std::move(func)](ui32 idx) {
            return func(realObjIdx[idx]);
        };
    }

    const size_t blockSize = Max<size_t>(CeilDiv<size_t>(docsSubset.size(), localExecutor->GetThreadCount() + 1), 1000);
    const TSimpleIndexRangesGenerator<size_t> rangesGenerator(TIndexRange<size_t>(docsSubset.size()), blockSize);
    const int blockCount = rangesGenerator.RangesCount();
    TConstArrayRef<ui32> docsSubsetRef(docsSubset);
    localExecutor->ExecRange(
        [&node, indicesRef, splitFunction, docsSubsetRef, &rangesGenerator](int blockId) {
            for (auto idx : rangesGenerator.GetRange(blockId).Iter()) {
                const int objIdx = docsSubsetRef[idx];
                indicesRef[docsSubsetRef[idx]] = (~node.Left) + splitFunction(objIdx) * ((~node.Right) - (~node.Left));
            }
        },
        0,
        blockCount,
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}

void UpdateIndicesWithSplit(
    const TSplitNode& node,
    const TTrainingDataProviders& trainingData,
    const NCB::TIndexedSubset<ui32>& docsSubset,
    const TFold& fold,
    NPar::ILocalExecutor* localExecutor,
    TArrayRef<TIndexType> indicesRef,
    NCB::TIndexedSubset<ui32>* leftIndices,
    NCB::TIndexedSubset<ui32>* rightIndices
) {
    const ui32 docSize = docsSubset.size();
    const ui32* docsSubsetPtr = docsSubset.data();
    TQuantizedObjectsDataProviderPtr objectsDataProvider;
    TIndexedSubsetCache indexedSubsetCache; // not really updated because it is used for test only
    const ui32* columnsIndexing = nullptr;
    GetObjectsDataAndIndexing(
        trainingData,
        fold,
        node.Split.Type == ESplitType::EstimatedFeature,
        node.Split.IsOnline(),
        /*objectSubsetIdx*/ 0, // 0 - learn
        &indexedSubsetCache,
        localExecutor,
        &objectsDataProvider,
        &columnsIndexing // can return nullptr
    );

    auto func = BuildNodeSplitFunction(
        node,
        *objectsDataProvider,
        node.Split.Type == ESplitType::OnlineCtr ? &fold.GetCtrs(node.Split.Ctr.Projection) : nullptr,
        /* docOffset */ 0);

    // TODO(ilyzhin) std::function is very slow for calling many times (maybe replace it with lambda)
    std::function<bool(ui32)> splitFunction;
    if (!columnsIndexing) {
        splitFunction = std::move(func);
    } else {
        splitFunction = [realObjIdx = columnsIndexing, func=std::move(func)](ui32 idx) {
            return func(realObjIdx[idx]);
        };
    }
    const size_t blockSize = Max<size_t>(CeilDiv<size_t>(docSize, localExecutor->GetThreadCount() + 1), 1000);
    const TSimpleIndexRangesGenerator<size_t> rangesGenerator(TIndexRange<size_t>(docSize), blockSize);
    const int blockCount = rangesGenerator.RangesCount();
    TVector<size_t> leftCounts(blockCount + 1, 0);
    TVector<size_t> rightCounts(blockCount + 1, 0);
    TVector<TVector<ui32>> localLefts(blockCount);
    TVector<TVector<ui32>> localRights(blockCount);

    localExecutor->ExecRange(
        [&node, indicesRef, splitFunction, &docsSubsetPtr, &rangesGenerator, &leftCounts, &rightCounts, &localLefts, &localRights](int blockId) {
            const ui32* docsSubsetLocal = docsSubsetPtr;
            const size_t rangeSize = rangesGenerator.GetRange(blockId).GetSize();
            localLefts[blockId].yresize(rangeSize);
            localRights[blockId].yresize(rangeSize);

            TArrayRef<ui32> localLeftsRef(localLefts[blockId]);
            TArrayRef<ui32> localRightsRef(localRights[blockId]);
            size_t nLeftCount = 0;
            size_t nRightCount = 0;
            for (auto idx : rangesGenerator.GetRange(blockId).Iter()) {
                const ui32 objIdx = *(docsSubsetLocal + idx);
                const bool split = splitFunction(objIdx);
                indicesRef[objIdx] = (~node.Left) + split * ((~node.Right) - (~node.Left));
                if (split) {
                    localRightsRef[nRightCount] = objIdx;
                    ++nRightCount;
                } else {
                    localLeftsRef[nLeftCount] = objIdx;
                    ++nLeftCount;
                }
            }
            leftCounts[blockId + 1] = nLeftCount;
            rightCounts[blockId + 1] = nRightCount;
        },
        0,
        blockCount,
        NPar::TLocalExecutor::WAIT_COMPLETE
    );

    for (int i = 1; i < blockCount + 1; ++i) {
        leftCounts[i] += leftCounts[i - 1];
        rightCounts[i] += rightCounts[i - 1];
    }

    const size_t nLeftCount = leftCounts[blockCount];
    const size_t nRightCount = rightCounts[blockCount];

    leftIndices->yresize(nLeftCount);
    rightIndices->yresize(nRightCount);
    TArrayRef<ui32> leftIndicesRef(*leftIndices);
    TArrayRef<ui32> rightIndicesRef(*rightIndices);

    localExecutor->ExecRange(
        [leftIndicesRef, rightIndicesRef, &leftCounts, &rightCounts, &localLefts, &localRights] (int blockId) {
            size_t leftStart = leftCounts[blockId];
            size_t rightStart = rightCounts[blockId];
            const size_t leftSize = leftCounts[blockId + 1] - leftStart;
            const size_t rightSize = rightCounts[blockId + 1] - rightStart;
            TConstArrayRef<ui32> localLeftsRef(localLefts[blockId].data(), leftSize);
            TConstArrayRef<ui32> localRightsRef(localRights[blockId].data(), rightSize);
            for (size_t j = 0; j < leftSize; ++j) {
                leftIndicesRef[leftStart++] = localLeftsRef[j];
            }
            for (size_t j = 0; j < rightSize; ++j) {
                rightIndicesRef[rightStart++] = localRightsRef[j];
            }
        },
        0,
        blockCount,
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}

void BuildIndicesForDataset(
    const TNonSymmetricTreeStructure& tree,
    const TTrainingDataProviders& trainingData,
    const TFold& fold,
    ui32 sampleCount,
    const TVector<const TOnlineCtrBase*>& onlineCtrs,
    ui32 objectSubsetIdx, // 0 - learn, 1+ - test (subtract 1 for testIndex)
    NPar::ILocalExecutor* localExecutor,
    TIndexType* indices) {

    TIndexedSubsetCache indexedSubsetCache;

    TConstArrayRef<TSplitNode> nodesRef = tree.GetNodes();

    TVector<std::function<bool(ui32 objIdx)>> nodesSplitFunctions;
    nodesSplitFunctions.yresize(tree.GetNodesCount());
    for (auto nodeIdx : xrange(tree.GetNodesCount())) {
        TQuantizedObjectsDataProviderPtr objectsDataProvider;
        const ui32* columnIndexing;
        GetObjectsDataAndIndexing(
            trainingData,
            fold,
            nodesRef[nodeIdx].Split.Type == ESplitType::EstimatedFeature,
            nodesRef[nodeIdx].Split.IsOnline(),
            objectSubsetIdx,
            &indexedSubsetCache,
            localExecutor,
            &objectsDataProvider,
            &columnIndexing);

        auto func = BuildNodeSplitFunction(
            nodesRef[nodeIdx],
            *objectsDataProvider,
            onlineCtrs[nodeIdx],
            objectSubsetIdx);
        if ((nodesRef[nodeIdx].Split.Type == ESplitType::OnlineCtr) || !columnIndexing) {
            nodesSplitFunctions[nodeIdx] = std::move(func);
        } else {
            nodesSplitFunctions[nodeIdx] = [realObjIdx = columnIndexing, func](ui32 idx) {
                return func(realObjIdx[idx]);
            };
        }
    }

    TConstArrayRef<std::function<bool(ui32 objIdx)>> nodesSplitFunctionsRef = nodesSplitFunctions;
    TArrayRef<TIndexType> indicesRef(indices, sampleCount);

    NPar::TLocalExecutor::TExecRangeParams params(0, sampleCount);
    params.SetBlockCount(localExecutor->GetThreadCount() + 1);
    localExecutor->ExecRange(
        [root=tree.GetRoot(), nodesRef, nodesSplitFunctionsRef, indicesRef](ui32 idx) {
            int nodeIdx = root;
            while (nodeIdx >= 0) {
                const auto& node = nodesRef[nodeIdx];
                nodeIdx = node.Left + nodesSplitFunctionsRef[nodeIdx](idx) * (node.Right - node.Left);
            }
            indicesRef[idx] = ~nodeIdx;
        },
        params,
        NPar::TLocalExecutor::WAIT_COMPLETE);
}
