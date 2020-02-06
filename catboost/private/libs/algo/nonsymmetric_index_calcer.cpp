#include "nonsymmetric_index_calcer.h"

#include "fold.h"
#include "online_ctr.h"
#include "split.h"

#include <catboost/libs/data/objects.h>
#include <catboost/libs/helpers/array_subset.h>

#include <util/generic/vector.h>

using namespace NCB;


static TConstArrayRef<ui8> GetCtrValues(const TSplit& split, const TOnlineCTR& ctr) {
    return ctr.Feature[split.Ctr.CtrIdx][split.Ctr.TargetBorderIdx][split.Ctr.PriorIdx];
}

template <typename T, EFeatureValuesType FeatureValuesType, class TCmpOp>
inline std::function<bool(ui32)> BuildNodeSplitFunction(
    const TTypedFeatureValuesHolder<T, FeatureValuesType>& column,
    TCmpOp cmpOp) {

    if (const auto* columnData
        = dynamic_cast<const TCompressedValuesHolderImpl<T, FeatureValuesType>*>(&column))
    {
        const TCompressedArray* compressedArray = columnData->GetCompressedData().GetSrc();

        std::function<bool(ui32)> func;
        NCB::DispatchBitsPerKeyToDataType(
            *compressedArray,
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

template <typename T, EFeatureValuesType FeatureValuesType, class TCmpOp>
std::function<bool(ui32)> BuildNodeSplitFunction(
    TMaybe<TExclusiveBundleIndex> maybeExclusiveBundleIndex,
    TMaybe<TPackedBinaryIndex> maybeBinaryIndex,
    TMaybe<TFeaturesGroupIndex> maybeFeaturesGroupIndex,
    TConstArrayRef<TExclusiveFeaturesBundle> exclusiveFeaturesBundlesMetaData,
    const TTypedFeatureValuesHolder<T, FeatureValuesType>& column,
    std::function<const TExclusiveFeatureBundleHolder*(ui32)>&& getExclusiveFeaturesBundle,
    std::function<const TBinaryPacksHolder*(ui32)>&& getBinaryFeaturesPack,
    std::function<const TFeaturesGroupHolder*(ui32)>&& getFeaturesGroup,
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
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const TOnlineCTR* onlineCtr,
    ui32 docOffset) {

    const auto& split = node.Split;

    if (split.Type == ESplitType::OnlineCtr) {
        const auto ctr = onlineCtr;
        const auto ctrValuesData = GetCtrValues(split, *ctr).data() + docOffset;
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


        if (split.Type == ESplitType::FloatFeature) {
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
    const NCB::TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const NCB::TIndexedSubset<ui32>& docsSubset,
    const TFold& fold,
    NPar::TLocalExecutor* localExecutor,
    TArrayRef<TIndexType> indicesRef
) {
    const auto* columnsIndexing
        = GetIf<TIndexedSubset<ui32>>(&fold.LearnPermutationFeaturesSubset);
    Y_ASSERT(columnsIndexing != nullptr);

    auto func = BuildNodeSplitFunction(
        node,
        objectsDataProvider,
        node.Split.Type == ESplitType::OnlineCtr ? &fold.GetCtr(node.Split.Ctr.Projection) : nullptr,
        /* docOffset */ 0);

    // TODO(ilyzhin) std::function is very slow for calling many times (maybe replace it with lambda)
    std::function<bool(ui32)> splitFunction;
    if (node.Split.Type == ESplitType::OnlineCtr) {
        splitFunction = std::move(func);
    } else {
        splitFunction = [realObjIdx = columnsIndexing->data(), func=std::move(func)](ui32 idx) {
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

void BuildIndicesForDataset(
    const TNonSymmetricTreeStructure& tree,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const NCB::TFeaturesArraySubsetIndexing& featuresArraySubsetIndexing,
    ui32 sampleCount,
    const TVector<const TOnlineCTR*>& onlineCtrs,
    ui32 docOffset,
    NPar::TLocalExecutor* localExecutor,
    TIndexType* indices) {

    TIndexedSubset<ui32> columnsIndexingStorage;

    const auto* columnsIndexing
        = GetIf<TIndexedSubset<ui32>>(&featuresArraySubsetIndexing);

    if (!columnsIndexing) {
        columnsIndexingStorage.yresize(featuresArraySubsetIndexing.Size());
        featuresArraySubsetIndexing.ParallelForEach(
            [columnsIndexingData = columnsIndexingStorage.data()](ui32 idx, ui32 srcIdx) {
                columnsIndexingData[idx] = srcIdx;
            },
            localExecutor);
        columnsIndexing = &columnsIndexingStorage;
    }

    TConstArrayRef<TSplitNode> nodesRef = tree.GetNodes();

    TVector<std::function<bool(ui32 objIdx)>> nodesSplitFunctions;
    nodesSplitFunctions.yresize(tree.GetNodesCount());
    for (auto nodeIdx : xrange(tree.GetNodesCount())) {
        auto func = BuildNodeSplitFunction(
            nodesRef[nodeIdx],
            objectsDataProvider,
            onlineCtrs[nodeIdx],
            docOffset);
        if (nodesRef[nodeIdx].Split.Type == ESplitType::OnlineCtr) {
            nodesSplitFunctions[nodeIdx] = std::move(func);
        } else {
            nodesSplitFunctions[nodeIdx] = [realObjIdx = columnsIndexing->data(), func](ui32 idx) {
                return func(realObjIdx[idx]);
            };
        }
    }

    TConstArrayRef<std::function<bool(ui32 objIdx)>> nodesSplitFunctionsRef = nodesSplitFunctions;
    TArrayRef<TIndexType> indicesRef(indices, sampleCount);

    localExecutor->ExecRange(
        [root=tree.GetRoot(), nodesRef, nodesSplitFunctionsRef, indicesRef](ui32 idx) {
            int nodeIdx = root;
            while (nodeIdx >= 0) {
                const auto& node = nodesRef[nodeIdx];
                nodeIdx = node.Left + nodesSplitFunctionsRef[nodeIdx](idx) * (node.Right - node.Left);
            }
            indicesRef[idx] = ~nodeIdx;
        },
        0,
        sampleCount,
        NPar::TLocalExecutor::WAIT_COMPLETE);
}
