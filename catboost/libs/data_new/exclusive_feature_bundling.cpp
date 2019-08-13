#include "exclusive_feature_bundling.h"

#include "packed_binary_features.h"
#include "objects.h"
#include "quantization.h"
#include "quantized_features_info.h"

#include <catboost/libs/helpers/dbg_output.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/options/restrictions.h>
#include <catboost/libs/helpers/vec_list.h>

#include <library/dbg_output/dump.h>
#include <library/pop_count/popcount.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/map.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/xrange.h>
#include <util/system/compiler.h>
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

        static inline ui32 IncrementCount(
            ui32 flatFeatureIdx2,
            ui32 intersectionCount,
            THashMap<ui32, ui32>* intersectionCountsForFeature1
        ) {
            THashMap<ui32, ui32>::insert_ctx insertCtx;

            auto it = intersectionCountsForFeature1->find(flatFeatureIdx2, insertCtx);
            if (it == intersectionCountsForFeature1->end()) {
                intersectionCountsForFeature1->emplace_direct(insertCtx, flatFeatureIdx2, intersectionCount);
                return intersectionCount;
            } else {
                it->second += intersectionCount;
                return it->second;
            }
        }

        ui32 IncrementCount(ui32 flatFeatureIdx1, ui32 flatFeatureIdx2, ui32 intersectionCount = 1) {
            if (SymmetricalIndices) {
                Y_ASSERT(flatFeatureIdx2 < IntersectionCounts.size());
            }
            return IncrementCount(flatFeatureIdx2, intersectionCount, &IntersectionCounts[flatFeatureIdx1]);
        }

        static inline void Add(
            const THashMap<ui32, ui32>& addIntersectionCountsForFeature1,
            THashMap<ui32, ui32>* dstIntersectionCountsForFeature1
        ) {
            for (const auto& [flatFeatureIdx2, intersectionCount] : addIntersectionCountsForFeature1) {
                IncrementCount(flatFeatureIdx2, intersectionCount, dstIntersectionCountsForFeature1);
            }
        }

        void Add(const TFeatureIntersectionGraph& rhs) {
            CB_ENSURE_INTERNAL(
                IntersectionCounts.size() == rhs.IntersectionCounts.size(),
                "Incompatible sizes of IntersectionCounts"
            );

            for (auto flatFeatureIdx1 : xrange(rhs.IntersectionCounts.size())) {
                Add(rhs.IntersectionCounts[flatFeatureIdx1], &IntersectionCounts[flatFeatureIdx1]);
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

    struct TCalcIntersectionCheckList {
        ui32 FeatureIdx1 = 0;
        TVecList<ui32> FeatureIndices2;

    public:
        TCalcIntersectionCheckList(ui32 featureIdx1)
            : FeatureIdx1(featureIdx1)
        {}
    };

    struct TToCalcData {
        ui32 CountersCount = 0;
    };

    // per-thread data
    struct TCalcFeatureIntersectionPartData {
        TFeatureIntersectionGraph FeatureIntersectionGraph;
        TVector<ui32> FeatureNonDefaultCount; // [flatFeatureIdx]

        TVecList<TCalcIntersectionCheckList> FeatureIntersectionCheckLists;
        TVecList<ui32> FeatureIndicesToCalc; // contains [flatFeatureIdx]
        TVector<TToCalcData> ToCalcStats; // [flatFeatureIdx]

        // updated for objects in CalcNonDefaultValuesMasks
        TVector<ui64> FeatureNonDefaultValuesMasks; // [flatFeatureIdx]

    public:
        TCalcFeatureIntersectionPartData()
            : FeatureIntersectionGraph(true)
        {}

        void Init(
            ui32 featureCount,
            const TVecList<TCalcIntersectionCheckList>& featureIntersectionCheckLists,
            const TVecList<ui32>& featureIndicesToCalc
        ) {
            FeatureIntersectionGraph.IntersectionCounts.resize(featureCount);
            FeatureIntersectionCheckLists = featureIntersectionCheckLists;
            FeatureIndicesToCalc = featureIndicesToCalc;

            ToCalcStats.resize(featureCount);
            for (auto featureIndicesToCalcIter = FeatureIndicesToCalc.begin();
                 featureIndicesToCalcIter != FeatureIndicesToCalc.end();
                 ++featureIndicesToCalcIter)
            {
                ToCalcStats[*featureIndicesToCalcIter].CountersCount = (ui32)FeatureIndicesToCalc.size() - 1;
            }

            FeatureNonDefaultValuesMasks.yresize(featureCount);
            FeatureNonDefaultCount.resize(featureCount, 0);
        }

        bool Inited() const {
            return !FeatureNonDefaultCount.empty();
        }

        void CalcNonDefaultValuesMasks(
            TConstArrayRef<TGetNonDefaultValuesMask> getNonDefaultValuesMaskFunctions,
            TConstArrayRef<ui32> srcIndices
        ) {
            for (auto featureIndicesToCalcIter = FeatureIndicesToCalc.begin();
                 featureIndicesToCalcIter != FeatureIndicesToCalc.end();)
            {
                auto flatFeatureIdx = *featureIndicesToCalcIter;
                if (!ToCalcStats[flatFeatureIdx].CountersCount) {
                    featureIndicesToCalcIter = FeatureIndicesToCalc.erase(featureIndicesToCalcIter);
                } else {
                    FeatureNonDefaultValuesMasks[flatFeatureIdx]
                        = getNonDefaultValuesMaskFunctions[flatFeatureIdx](srcIndices);
                    ++featureIndicesToCalcIter;
                }
            }
        }

        void UpdateNonDefaultCounts(
            TConstArrayRef<TGetNonDefaultValuesMask> getNonDefaultValuesMaskFunctions,
            TConstArrayRef<ui32> srcIndices
        ) {
            for (auto featureIndicesToCalcIter = FeatureIndicesToCalc.begin();
                 featureIndicesToCalcIter != FeatureIndicesToCalc.end();)
            {
                auto flatFeatureIdx = *featureIndicesToCalcIter;
                if (!ToCalcStats[flatFeatureIdx].CountersCount) {
                    featureIndicesToCalcIter = FeatureIndicesToCalc.erase(featureIndicesToCalcIter);
                } else {
                    FeatureNonDefaultCount[flatFeatureIdx]
                        += PopCount(getNonDefaultValuesMaskFunctions[flatFeatureIdx](srcIndices));
                    ++featureIndicesToCalcIter;
                }
            }
        }

        /* TCheckAndUpdateForPairFunction must be of type bool (featureIdx1, featureIdx2) and return true if
         * no more intersection count calculation for the pair is required for this part
         */
        template <class TCheckSkipFeatureFunction, class TCheckAndUpdateForPairFunction>
        void CheckAndUpdate(
            TCheckSkipFeatureFunction&& checkSkipFeatureFunction,
            TCheckAndUpdateForPairFunction&& checkAndUpdateForPairFunction
        ) {
            auto featureIntersectionCheckListsIter = FeatureIntersectionCheckLists.begin();
            while (featureIntersectionCheckListsIter != FeatureIntersectionCheckLists.end()) {
                auto& featureIntersectionCheckList = *featureIntersectionCheckListsIter;
                auto featureIdx1 = featureIntersectionCheckList.FeatureIdx1;

                if (!checkSkipFeatureFunction(featureIdx1)) {
                    auto& featureIndices2 = featureIntersectionCheckList.FeatureIndices2;
                    auto indices2Iter = featureIndices2.begin();
                    while (indices2Iter != featureIndices2.end()) {
                        auto featureIdx2 = *indices2Iter;
                        if (checkAndUpdateForPairFunction(featureIdx1, featureIdx2)) {
                            auto& toCalcStatsForIdx2 = ToCalcStats[featureIdx2];
                            Y_ASSERT(toCalcStatsForIdx2.CountersCount > 0);
                            --toCalcStatsForIdx2.CountersCount;
                            auto& toCalcStatsForIdx1 = ToCalcStats[featureIdx1];
                            Y_ASSERT(toCalcStatsForIdx1.CountersCount > 0);
                           --toCalcStatsForIdx1.CountersCount;

                            indices2Iter = featureIndices2.erase(indices2Iter);
                            if (featureIndices2.empty()) {
                                featureIntersectionCheckListsIter
                                    = FeatureIntersectionCheckLists.erase(
                                        featureIntersectionCheckListsIter
                                    );
                                goto next_outer_loop_iter;
                            }
                        } else {
                            ++indices2Iter;
                        }
                    }
                }
                ++featureIntersectionCheckListsIter;
next_outer_loop_iter:
                ;
            }
        }
    };

    struct TFeatureWithDegree {
        ui32 FlatFeatureIdx;
        ui32 Degree;

    public:
        bool operator > (const TFeatureWithDegree& rhs) const {
            return Degree > rhs.Degree;
        }
    };


    static TVector<TExclusiveFeaturesBundle> CreateExclusiveFeatureBundlesFromGraph(
        const TFeaturesLayout& featuresLayout,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        TFeatureIntersectionGraph&& featureIntersectionGraph,
        ui32 objectCount,
        const TExclusiveFeaturesBundlingOptions& options
    ) {
        featureIntersectionGraph.MakeSymmetric();

        const auto featuresMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo();


        TVector<TFeatureWithDegree> featuresWithDegree;

        for (auto flatFeatureIdx : xrange(featuresLayout.GetExternalFeatureCount())) {
            if (!featuresMetaInfo[flatFeatureIdx].IsAvailable ||
                featuresMetaInfo[flatFeatureIdx].Type == EFeatureType::Text) {
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


    void MergeParts(
        ui32 featureCount,
        TConstArrayRef<ui32> featureIndicesToCalc,
        bool canMove,
        TArrayRef<TCalcFeatureIntersectionPartData> partsData,
        NPar::TLocalExecutor* localExecutor,
        TFeatureIntersectionGraph* dst
    ) {
        dst->IntersectionCounts.resize(featureCount);
        localExecutor->ExecRange(
            [&] (int featureToCalcIdx) {
                const ui32 featureIdx1 = featureIndicesToCalc[featureToCalcIdx];

                auto* intersectionCountsForFeature1 = &(dst->IntersectionCounts[featureIdx1]);

                auto& part0Counts = partsData[0].FeatureIntersectionGraph.IntersectionCounts[featureIdx1];
                if (canMove) {
                    *intersectionCountsForFeature1 = std::move(part0Counts);
                } else {
                    *intersectionCountsForFeature1 = part0Counts;
                }

                for (auto partIdx : xrange(size_t(1), partsData.size())) {
                    TFeatureIntersectionGraph::Add(
                        partsData[partIdx].FeatureIntersectionGraph.IntersectionCounts[featureIdx1],
                        intersectionCountsForFeature1
                    );
                }
            },
            0,
            SafeIntegerCast<int>(featureIndicesToCalc.size()),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }


    TVector<TExclusiveFeaturesBundle> CreateExclusiveFeatureBundles(
        const TRawObjectsData& rawObjectsData,
        const TFeaturesArraySubsetIndexing& rawDataSubsetIndexing,
        const TFeaturesLayout& featuresLayout,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TExclusiveFeaturesBundlingOptions& options,
        NPar::TLocalExecutor* localExecutor
    ) {
        const ui32 objectCount = rawDataSubsetIndexing.Size();

        if (objectCount == 0) {
            return {};
        }

        const auto featureCount = featuresLayout.GetExternalFeatureCount();
        const auto featuresMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo();

        TVecList<TCalcIntersectionCheckList> featureIntersectionCheckLists;
        TVecList<ui32> featureIndicesToCalc;
        TVector<ui32> featureIndicesToCalcVector; // needed for random-access during parallel execution
        TVector<TGetNonDefaultValuesMask> getNonDefaultValuesMaskFunctions(featureCount); // [flatFeatureIdx]

        for (auto flatFeatureIdx : xrange(featureCount)) {
            const auto& featureMetaInfo = featuresMetaInfo[flatFeatureIdx];
            if (!featureMetaInfo.IsAvailable || featureMetaInfo.Type == EFeatureType::Text) {
                continue;
            }

            // add previous element. Algorithm works this way to avoid adding the last available feature.
            if (!featureIndicesToCalcVector.empty()) {
                featureIntersectionCheckLists.push_back(
                    TCalcIntersectionCheckList(featureIndicesToCalcVector.back())
                );
                for (auto& featureIntersectionCheckList : featureIntersectionCheckLists) {
                    featureIntersectionCheckList.FeatureIndices2.push_back(flatFeatureIdx);
                }
            }

            featureIndicesToCalc.push_back(flatFeatureIdx);
            featureIndicesToCalcVector.push_back(flatFeatureIdx);

            if (featureMetaInfo.Type == EFeatureType::Float) {
                getNonDefaultValuesMaskFunctions[flatFeatureIdx] = GetQuantizedFloatNonDefaultValuesMaskFunction(
                    rawObjectsData,
                    quantizedFeaturesInfo,
                    TFloatFeatureIdx(featuresLayout.GetInternalFeatureIdx(flatFeatureIdx))
                );
            } else if (featureMetaInfo.Type == EFeatureType::Categorical) {
                getNonDefaultValuesMaskFunctions[flatFeatureIdx] = GetQuantizedCatNonDefaultValuesMaskFunction(
                    rawObjectsData,
                    quantizedFeaturesInfo,
                    TCatFeatureIdx(featuresLayout.GetInternalFeatureIdx(flatFeatureIdx))
                );
            } else {
                CB_ENSURE(false, featureMetaInfo.Type << " is not supported for feature bundles");
            }
        }

        TVector<ui32> subsetIndices;
        subsetIndices.yresize(rawDataSubsetIndexing.Size());

        rawDataSubsetIndexing.ParallelForEach(
            [&] (ui32 objectIdx, ui32 srcObjectIdx) {
                subsetIndices[objectIdx] = srcObjectIdx;
            },
            localExecutor
        );

        // to improve locality
        Sort(subsetIndices);

        TSimpleIndexRangesGenerator<ui32> partRanges(
            TIndexRange<ui32>(0, objectCount),
            CeilDiv(objectCount, ui32(localExecutor->GetThreadCount() + 1))
        );

        const int partCount = partRanges.RangesCount();

        TVector<TCalcFeatureIntersectionPartData> partsData(partCount);

        const ui32 maxObjectIntersection = ui32(options.MaxConflictFraction * float(objectCount));


        localExecutor->ExecRange(
            [&](int partIdx) {
                auto& part = partsData[partIdx];
                part.Init(featureCount, featureIntersectionCheckLists, featureIndicesToCalc);

                auto partObjectRange = partRanges.GetRange(partIdx);

                TSimpleIndexRangesGenerator<ui32> partBlocks(partObjectRange, (ui32)sizeof(ui64) * CHAR_BIT);

                for (auto blockIdx : xrange(partBlocks.RangesCount())) {
                    auto blockRange = partBlocks.GetRange(blockIdx);

                    part.UpdateNonDefaultCounts(
                        getNonDefaultValuesMaskFunctions,
                        TConstArrayRef<ui32>(subsetIndices.begin() + blockRange.Begin, blockRange.GetSize())
                    );
                }


                part.CheckAndUpdate(
                    /*checkSkipFeatureFunction*/ [&] (ui32 featureIdx1) -> bool {
                        Y_UNUSED(featureIdx1);
                        return false;
                    },
                    /*checkAndUpdateForPairFunction*/ [&] (ui32 featureIdx1, ui32 featureIdx2) -> bool {
                        const auto nonDefaultCount1 = part.FeatureNonDefaultCount[featureIdx1];
                        if (nonDefaultCount1 == 0) {
                            return true;
                        }

                        const auto nonDefaultCount2 = part.FeatureNonDefaultCount[featureIdx2];

                        if (nonDefaultCount1 == partObjectRange.GetSize()) {
                            if (nonDefaultCount2) {
                                part.FeatureIntersectionGraph.IncrementCount(
                                    featureIdx1,
                                    featureIdx2,
                                    nonDefaultCount2
                                );
                            }
                            return true;
                        } else if ((nonDefaultCount1 + nonDefaultCount2)
                                   > (partObjectRange.GetSize() + maxObjectIntersection))
                        {
                            part.FeatureIntersectionGraph.IncrementCount(
                                featureIdx1,
                                featureIdx2,
                                maxObjectIntersection + 1 // any value over maxObjectIntersection will do
                            );
                            return true;
                        } else {
                            return false;
                        }
                    }
                );

            },
            0,
            partCount,
            NPar::TLocalExecutor::WAIT_COMPLETE
        );

        TFeatureIntersectionGraph featureIntersectionGraphAfterStage1(true);

        MergeParts(
            featureCount,
            featureIndicesToCalcVector,
            /*canMove*/ false,
            partsData,
            localExecutor,
            &featureIntersectionGraphAfterStage1
        );

        localExecutor->ExecRange(
            [&](int partIdx) {
                auto& part = partsData[partIdx];

                // remove counters not needed after stage1

                part.CheckAndUpdate(
                    /*checkSkipFeatureFunction*/ [&] (ui32 featureIdx1) -> bool {
                        Y_UNUSED(featureIdx1);
                        return false;
                    },
                    /*checkAndUpdateForPairFunction*/ [&] (ui32 featureIdx1, ui32 featureIdx2) -> bool {
                        const auto* intersectionCountPtr = MapFindPtr(
                            featureIntersectionGraphAfterStage1.IntersectionCounts[featureIdx1],
                            featureIdx2
                        );

                        return intersectionCountPtr && (*intersectionCountPtr > maxObjectIntersection);
                    }
                );

                if (part.FeatureIndicesToCalc.empty()) {
                    return;
                }

                auto partObjectRange = partRanges.GetRange(partIdx);

                TSimpleIndexRangesGenerator<ui32> partBlocks(partObjectRange, (ui32)sizeof(ui64) * CHAR_BIT);

                for (auto blockIdx : xrange(partBlocks.RangesCount())) {
                    auto blockRange = partBlocks.GetRange(blockIdx);

                    part.CalcNonDefaultValuesMasks(
                        getNonDefaultValuesMaskFunctions,
                        TConstArrayRef<ui32>(subsetIndices.begin() + blockRange.Begin, blockRange.GetSize())
                    );

                    part.CheckAndUpdate(
                        /*checkSkipFeatureFunction*/ [&] (ui32 featureIdx1) -> bool {
                            return !part.FeatureNonDefaultValuesMasks[featureIdx1];
                        },
                        /*checkAndUpdateForPairFunction*/ [&] (ui32 featureIdx1, ui32 featureIdx2) -> bool {
                            const ui32 intersectionCount = (ui32)PopCount(
                                part.FeatureNonDefaultValuesMasks[featureIdx1] &
                                part.FeatureNonDefaultValuesMasks[featureIdx2]
                            );
                            if (!intersectionCount) {
                                return false;
                            }
                            return part.FeatureIntersectionGraph.IncrementCount(
                                featureIdx1,
                                featureIdx2
                            ) > maxObjectIntersection;
                        }
                    );
                }
            },
            0,
            partCount,
            NPar::TLocalExecutor::WAIT_COMPLETE
        );

        featureIntersectionGraphAfterStage1.IntersectionCounts.clear(); // free memory

        TFeatureIntersectionGraph resultFeatureIntersectionGraph(true);

        MergeParts(
            featureCount,
            featureIndicesToCalcVector,
            /*canMove*/ true,
            partsData,
            localExecutor,
            &resultFeatureIntersectionGraph
        );

        return CreateExclusiveFeatureBundlesFromGraph(
            featuresLayout,
            quantizedFeaturesInfo,
            std::move(resultFeatureIntersectionGraph),
            rawDataSubsetIndexing.Size(),
            options
        );
    }

}
