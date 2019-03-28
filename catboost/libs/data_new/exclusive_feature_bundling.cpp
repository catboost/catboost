#include "exclusive_feature_bundling.h"

#include "packed_binary_features.h"
#include "objects.h"
#include "quantization.h"
#include "quantized_features_info.h"

#include <catboost/libs/helpers/dbg_output.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/options/restrictions.h>

#include <library/dbg_output/dump.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/bitops.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/map.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/system/yassert.h>

#include <algorithm>
#include <array>
#include <functional>
#include <set>


namespace NCB {

    struct TFeatureIntersectionGraph {
        // flatFeatureIdx1 -> (flatFeatureIdx2 -> intersectionCount)
        TVector<THashMap<ui32, ui32>> IntersectionCounts;

        bool SymmetricalIndices; // feature indices are the same

    public:
        explicit TFeatureIntersectionGraph(bool symmetricalIndices)
            : SymmetricalIndices(symmetricalIndices)
        {}

        void IncrementCount(ui32 flatFeatureIdx1, ui32 flatFeatureIdx2, ui32 intersectionCount = 1) {
            if (SymmetricalIndices) {
                Y_ASSERT(flatFeatureIdx2 < IntersectionCounts.size());
            }

            auto& dstMap = IntersectionCounts[flatFeatureIdx1];

            THashMap<ui32, ui32>::insert_ctx insertCtx;

            auto dstMapIter = dstMap.find(flatFeatureIdx2, insertCtx);
            if (dstMapIter == dstMap.end()) {
                dstMap.emplace_direct(insertCtx, flatFeatureIdx2, intersectionCount);
            } else {
                dstMapIter->second += intersectionCount;
            }
        }

        void Add(const TFeatureIntersectionGraph& rhs) {
            CB_ENSURE_INTERNAL(
                IntersectionCounts.size() == rhs.IntersectionCounts.size(),
                "Incompatible sizes of IntersectionCounts"
            );

            for (auto flatFeatureIdx1 : xrange(rhs.IntersectionCounts.size())) {
                for (const auto& [flatFeatureIdx2, intersectionCount] :
                     rhs.IntersectionCounts[flatFeatureIdx1])
                {
                    IncrementCount(flatFeatureIdx1, flatFeatureIdx2, intersectionCount);
                }
            }
        }

        void MakeSymmetric() {
            CB_ENSURE_INTERNAL(
                SymmetricalIndices,
                "MakeSymmetric is applicable only for TFeatureIntersectionGraph with symmetric indices"
            );

            for (auto flatFeatureIdx1 : xrange(IntersectionCounts.size())) {
                for (const auto& [flatFeatureIdx2, intersectionCount] :
                     IntersectionCounts[flatFeatureIdx1])
                {
                    /* check because IntersectionCounts is changed in-place so some parts are updated
                        in process
                    */
                    if (flatFeatureIdx2 > flatFeatureIdx1) {
                        IncrementCount(flatFeatureIdx2, flatFeatureIdx1, intersectionCount);
                    }
                }
            }
        }

        ui32 GetDegree(ui32 flatFeatureIdx) const {
            return IntersectionCounts[flatFeatureIdx].size();
        }
    };

}

template <>
struct TDumper<NCB::TFeatureIntersectionGraph> {
    template <class S>
    static inline void Dump(S& s, const NCB::TFeatureIntersectionGraph& featureIntersectionGraph) {
        s << "SymmetricalIndices=" << featureIntersectionGraph.SymmetricalIndices << Endl;
        for (auto i1 : xrange(featureIntersectionGraph.IntersectionCounts.size())) {
            const auto& unorderedMap = featureIntersectionGraph.IntersectionCounts[i1];
            TMap<ui32, ui32> orderedMap(unorderedMap.begin(), unorderedMap.end());

            for (const auto& [i2, count] : orderedMap) {
                s << "IntersectionCount(" << i1 << "," << i2 << ")=" << count << Endl;
            }
        }
        s << Endl;
    }
};


namespace NCB {

    // per-thread data
    struct TCalcIntersectionPartData {
        TFeatureIntersectionGraph FeatureIntersectionGraph;
        TVector<ui32> QuantizedFeatureBins;

    public:
        TCalcIntersectionPartData()
            : FeatureIntersectionGraph(true)
        {}
    };

    static inline void CalcQuantizedFeatureBins(
        TConstArrayRef<TFeatureMetaInfo> featuresMetaInfo,
        TConstArrayRef<TGetBinFunction> getBinFunctions,
        ui32 idx,
        ui32 srcIdx,
        TVector<ui32>* quantizedFeatureBins
    ) {
        for (auto flatFeatureIdx : xrange(featuresMetaInfo.size())) {
            if (!featuresMetaInfo[flatFeatureIdx].IsAvailable) {
                continue;
            }

            (*quantizedFeatureBins)[flatFeatureIdx] = getBinFunctions[flatFeatureIdx](idx, srcIdx);
        }
    }

    struct TFeatureWithDegree {
        ui32 FlatFeatureIdx;
        ui32 Degree;

    public:
        bool operator > (const TFeatureWithDegree& rhs) const {
            return Degree > rhs.Degree;
        }
    };


    static TVector<TExclusiveFeaturesBundle> CreateExclusiveFeatureBundlesFromGraph(
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        TFeatureIntersectionGraph&& featureIntersectionGraph,
        ui32 objectCount,
        const TExclusiveFeaturesBundlingOptions& options
    ) {
        featureIntersectionGraph.MakeSymmetric();

        const auto& featuresLayout = *quantizedFeaturesInfo.GetFeaturesLayout();
        const auto featuresMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo();


        TVector<TFeatureWithDegree> featuresWithDegree;

        for (auto flatFeatureIdx : xrange(featuresLayout.GetExternalFeatureCount())) {
            if (!featuresMetaInfo[flatFeatureIdx].IsAvailable) {
                continue;
            }

            featuresWithDegree.push_back(
                TFeatureWithDegree{flatFeatureIdx, featureIntersectionGraph.GetDegree(flatFeatureIdx)}
            );
        }

        Sort(featuresWithDegree, std::greater<TFeatureWithDegree>());

        const ui32 maxObjectIntersection = ui32(options.MaxConflictFraction * float(objectCount));


        TVector<TExclusiveFeaturesBundle> resultOfStage1;

        TFeatureIntersectionGraph bundleToFeatureIntersectionGraph(false); // first index is bundle index

        auto bundlesGreaterByDegree = [&] (ui32 bundleIdx1, ui32 bundleIdx2) -> bool {
            return bundleToFeatureIntersectionGraph.GetDegree(bundleIdx1)
                > bundleToFeatureIntersectionGraph.GetDegree(bundleIdx2);
        };

        std::set<ui32, decltype(bundlesGreaterByDegree)> bundlesByDegree(bundlesGreaterByDegree);

        // shortcut for easier sequential one hot bundling
        TVector<TMaybe<ui32>> flatFeatureIdxToBundleIdx(featuresLayout.GetExternalFeatureCount());

        for (const auto& featureWithDegree : featuresWithDegree) {
            const auto flatFeatureIdx = featureWithDegree.FlatFeatureIdx;
            const auto featureType = featuresLayout.GetExternalFeatureType(flatFeatureIdx);
            const auto perTypeFeatureIdx = featuresLayout.GetInternalFeatureIdx(flatFeatureIdx);

            const ui32 featureBinCount
                = (featureType == EFeatureType::Float) ?
                    quantizedFeaturesInfo.GetBinCount(TFloatFeatureIdx(perTypeFeatureIdx)) :
                    quantizedFeaturesInfo.GetUniqueValuesCounts(TCatFeatureIdx(perTypeFeatureIdx)).OnAll;

            // because 0 bin is common for all features in the bundle
            const ui32 binCountInBundleNeeded = featureBinCount - 1;

            // returns true if features was added to this bundle
            auto tryAddToBundle = [&] (auto bundleIdx) -> bool {
                auto& bundle = resultOfStage1[bundleIdx];

                if ((bundle.GetBinCount() + binCountInBundleNeeded) > options.MaxBuckets) {
                    return false;
                }

                const auto* intersectionCountPtr = MapFindPtr(
                    bundleToFeatureIntersectionGraph.IntersectionCounts[bundleIdx],
                    flatFeatureIdx
                );
                if (intersectionCountPtr && (*intersectionCountPtr > maxObjectIntersection)) {
                    return false;
                }

                const ui32 lowerBoundInBundle = bundle.Parts.back().Bounds.End;
                const TBoundsInBundle boundsInBundle(
                    lowerBoundInBundle,
                    lowerBoundInBundle + binCountInBundleNeeded
                );

                bundlesByDegree.erase(bundleIdx);
                bundle.Add(TExclusiveBundlePart(featureType, perTypeFeatureIdx, boundsInBundle));
                bundlesByDegree.insert(bundleIdx);

                for (const auto& [flatFeatureIdx2, intersectionCount]
                     : featureIntersectionGraph.IntersectionCounts[flatFeatureIdx])
                {
                    bundleToFeatureIntersectionGraph.IncrementCount(
                        bundleIdx,
                        flatFeatureIdx2,
                        intersectionCount
                    );
                }
                flatFeatureIdxToBundleIdx[flatFeatureIdx] = bundleIdx;

                return true;
            };

            THashSet<ui32> checkedBundlesForNeighbors;

            // try neighboring bundles first
            if ((flatFeatureIdx > 0) && flatFeatureIdxToBundleIdx[flatFeatureIdx - 1].Defined()) {
                auto bundleIdx = *(flatFeatureIdxToBundleIdx[flatFeatureIdx - 1]);
                if (tryAddToBundle(bundleIdx)) {
                    continue;
                }
                checkedBundlesForNeighbors.insert(bundleIdx);
            }
            if (((flatFeatureIdx + 1) < flatFeatureIdxToBundleIdx.size()) &&
                flatFeatureIdxToBundleIdx[flatFeatureIdx + 1].Defined())
            {
                auto bundleIdx = *(flatFeatureIdxToBundleIdx[flatFeatureIdx + 1]);
                if (checkedBundlesForNeighbors.contains(bundleIdx)) {
                    continue;
                }
                if (tryAddToBundle(bundleIdx)) {
                    continue;
                }
                checkedBundlesForNeighbors.insert(bundleIdx);
            }

            // try
            bool bundleFound = false;
            for (auto bundleIdx : bundlesByDegree) {
                if (checkedBundlesForNeighbors.contains(bundleIdx)) {
                    continue;
                }
                if (tryAddToBundle(bundleIdx)) {
                    bundleFound = true;
                    break;
                }
            }

            // no existing bundle to add to - add new
            if (!bundleFound && (featureBinCount <= options.MaxBuckets)) {
                TExclusiveFeaturesBundle exclusiveFeaturesBundle;
                const TBoundsInBundle boundsInBundle(0, binCountInBundleNeeded);
                exclusiveFeaturesBundle.Add(
                    TExclusiveBundlePart(featureType, perTypeFeatureIdx, boundsInBundle)
                );
                resultOfStage1.push_back(std::move(exclusiveFeaturesBundle));
                auto bundleIdx = ui32(resultOfStage1.size() - 1);
                bundleToFeatureIntersectionGraph.IntersectionCounts.push_back(
                    featureIntersectionGraph.IntersectionCounts[flatFeatureIdx]
                );
                bundlesByDegree.insert(bundleIdx);
                flatFeatureIdxToBundleIdx[flatFeatureIdx] = bundleIdx;
            }
        }

        TVector<ui32> bundlesForResult;

        // less than sizeof(TBinaryFeaturesPack) * CHAR_BIT binary features
        TVector<ui32> smallBinaryFeaturesOnlyBundles;

        size_t binaryFeaturesInSmallBundlesCount = 0;

        for (auto i : xrange(resultOfStage1.size())) {
            auto partCount = resultOfStage1[i].Parts.size();

            // 2 is an empiric relative performance constant
            if (resultOfStage1[i].IsBinaryFeaturesOnly() &&
                (2 * partCount < (sizeof(TBinaryFeaturesPack) * CHAR_BIT)))
            {
                binaryFeaturesInSmallBundlesCount += partCount;
                smallBinaryFeaturesOnlyBundles.push_back(i);
            } else if (partCount > 1) { // don't save bundles with only one feature
                bundlesForResult.push_back(i);
            }
        }

        if (CeilDiv(binaryFeaturesInSmallBundlesCount, sizeof(TBinaryFeaturesPack)*CHAR_BIT)
            >= 2 * smallBinaryFeaturesOnlyBundles.size()) // 2 is an empiric relative performance constant
        {
            // no reason to repack bundles with binary features to packed binary histograms

            for (auto i : smallBinaryFeaturesOnlyBundles) {
                if (resultOfStage1[i].Parts.size() > 1) {
                    bundlesForResult.push_back(i);
                }
            }
        }

        if (bundlesForResult.size() == resultOfStage1.size()) {
            return resultOfStage1;
        }

        Sort(bundlesForResult);

        TVector<TExclusiveFeaturesBundle> result;
        result.reserve(bundlesForResult.size());

        for (auto stage1Idx : bundlesForResult) {
            result.push_back(std::move(resultOfStage1[stage1Idx]));
        }

        return result;
    }


    TVector<TExclusiveFeaturesBundle> CreateExclusiveFeatureBundles(
        const TRawObjectsData& rawObjectsData,
        const TFeaturesArraySubsetIndexing& rawDataSubsetIndexing,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TExclusiveFeaturesBundlingOptions& options,
        NPar::TLocalExecutor* localExecutor
    ) {
        const auto& featuresLayout = *quantizedFeaturesInfo.GetFeaturesLayout();

        const auto featureCount = featuresLayout.GetExternalFeatureCount();
        const auto featuresMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo();

        TVector<TGetBinFunction> getBinFunctions(featureCount); // [flatFeatureIdx]

        for (auto flatFeatureIdx : xrange(featureCount)) {
            const auto& featureMetaInfo = featuresMetaInfo[flatFeatureIdx];
            if (!featureMetaInfo.IsAvailable) {
                continue;
            }

            if (featureMetaInfo.Type == EFeatureType::Float) {
                getBinFunctions[flatFeatureIdx] = GetQuantizedFloatFeatureFunction(
                    rawObjectsData,
                    quantizedFeaturesInfo,
                    TFloatFeatureIdx(featuresLayout.GetInternalFeatureIdx(flatFeatureIdx))
                );
            } else {
                getBinFunctions[flatFeatureIdx] = GetQuantizedCatFeatureFunction(
                    rawObjectsData,
                    quantizedFeaturesInfo,
                    TCatFeatureIdx(featuresLayout.GetInternalFeatureIdx(flatFeatureIdx))
                );
            }
        }


        std::array<TCalcIntersectionPartData, CB_THREAD_LIMIT> partsData;

        for (auto& part : partsData) {
            part.FeatureIntersectionGraph.IntersectionCounts.resize(featureCount);
            part.QuantizedFeatureBins.resize(featureCount);
        }


        rawDataSubsetIndexing.ParallelForEach(
            [&] (ui32 idx, ui32 srcIdx) {
                int partIdx = localExecutor->GetWorkerThreadId();
                Y_ASSERT(partIdx < CB_THREAD_LIMIT);
                auto& part = partsData[partIdx];

                CalcQuantizedFeatureBins(
                    featuresMetaInfo,
                    getBinFunctions,
                    idx,
                    srcIdx,
                    &part.QuantizedFeatureBins
                );

                for (auto flatFeatureIdx1 : xrange(featureCount - 1)) {
                    const auto& featureMetaInfo1 = featuresMetaInfo[flatFeatureIdx1];
                    if (!featureMetaInfo1.IsAvailable) {
                        continue;
                    }

                    if (!part.QuantizedFeatureBins[flatFeatureIdx1]) {
                        continue;
                    }

                    for (auto flatFeatureIdx2 : xrange(flatFeatureIdx1 + 1, featureCount)) {
                        const auto& featureMetaInfo2 = featuresMetaInfo[flatFeatureIdx2];
                        if (!featureMetaInfo2.IsAvailable) {
                            continue;
                        }
                        if (part.QuantizedFeatureBins[flatFeatureIdx2]) {
                            part.FeatureIntersectionGraph.IncrementCount(flatFeatureIdx1, flatFeatureIdx2);
                        }
                    }
                }

            },
            localExecutor
        );

        // accumulate result in parts[0]
        auto& resultFeatureIntersectionGraph = partsData[0].FeatureIntersectionGraph;

        for (auto partIdx : xrange(size_t(1), partsData.size())) {
            resultFeatureIntersectionGraph.Add(partsData[partIdx].FeatureIntersectionGraph);
        }

        return CreateExclusiveFeatureBundlesFromGraph(
            quantizedFeaturesInfo,
            std::move(resultFeatureIntersectionGraph),
            rawDataSubsetIndexing.Size(),
            options
        );
    }

}
