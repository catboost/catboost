#include "exclusive_feature_bundling.h"

#include "packed_binary_features.h"
#include "quantization.h"
#include "quantized_features_info.h"

#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/double_array_iterator.h>
#include <catboost/libs/helpers/parallel_tasks.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/hash_set.h>
#include <util/generic/maybe.h>
#include <util/generic/utility.h>
#include <util/generic/xrange.h>
#include <util/random/fast.h>

#include <utility>
#include <bit>


namespace NCB {

    static_assert(CHAR_BIT == 8, "CatBoost requires CHAR_BIT == 8");

    // [flatFeatureIdx] -> (blockIdx, nonZero Mask)
    using TFeaturesNonDefaultMasks = TVector<TVector<std::pair<ui32, ui64>>>;

    struct TExclusiveFeatureBundleForMerging {
        ui32 IntersectionCount;
        ui64 NonDefaultCount;
        TVector<ui64> UsedObjects; // block non default masks

    public:
        TExclusiveFeatureBundleForMerging(ui32 objectCount, NPar::ILocalExecutor* localExecutor)
            : IntersectionCount(0)
            , NonDefaultCount(0)
        {
            UsedObjects.yresize(CeilDiv((size_t)objectCount, CHAR_BIT * sizeof(ui64)));
            ParallelFill<ui64>(0, /*blockSize*/ Nothing(), localExecutor, UsedObjects);
        }
    };


    static void AddFeatureToBundle(
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        ui32 flatFeatureIdx,
        TConstArrayRef<std::pair<ui32, ui64>> featureNonDefaultMasks,
        ui32 featureNonDefaultCount,
        ui32 binCountInBundleNeeded,
        ui32 intersectionCount,
        TExclusiveFeaturesBundle* bundle,
        TExclusiveFeatureBundleForMerging* bundleForMerging
    ) {
        const auto& featuresLayout = *quantizedFeaturesInfo.GetFeaturesLayout();
        const auto featureMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo()[flatFeatureIdx];

        TExclusiveBundlePart part;
        part.FeatureType = featureMetaInfo.Type;
        part.FeatureIdx = featuresLayout.GetInternalFeatureIdx(flatFeatureIdx);
        auto usedBinCount = bundle->GetUsedByPartsBinCount();

        part.Bounds = TBoundsInBundle(usedBinCount, usedBinCount + binCountInBundleNeeded);
        bundle->Add(std::move(part));

        bundleForMerging->NonDefaultCount += featureNonDefaultCount;
        bundleForMerging->IntersectionCount += intersectionCount;

        TArrayRef<ui64> bundleMasks = bundleForMerging->UsedObjects;

        for (auto [blockIdx, mask] : featureNonDefaultMasks) {
            bundleMasks[blockIdx] = bundleMasks[blockIdx] | mask;
        }
    }

    static ui32 CalcIntersectionCount(
        TConstArrayRef<ui64> usedObjectsInBundle,
        TConstArrayRef<std::pair<ui32, ui64>> featureNonDefaultMasks,
        ui32 maxIntersectionCount
    ) {
        ui32 intersectionCount = 0;
        for (auto [blockIdx, mask] : featureNonDefaultMasks) {
            intersectionCount += (ui32)std::popcount(mask & usedObjectsInBundle[blockIdx]);
            if (intersectionCount > maxIntersectionCount) {
                return intersectionCount;
            }
        }
        return intersectionCount;
    }

    static TVector<TExclusiveFeaturesBundle> CreateExclusiveFeatureBundlesImpl(
        ui32 objectCount,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TFeaturesNonDefaultMasks& featuresNonDefaultMasks,
        TConstArrayRef<ui32> featuresNonDefaultCounts,
        TConstArrayRef<ui32> flatFeatureIndicesToCalc,
        const TExclusiveFeaturesBundlingOptions& options,
        NPar::ILocalExecutor* localExecutor
    ) {
        const auto& featuresLayout = *quantizedFeaturesInfo.GetFeaturesLayout();

        const ui32 maxObjectIntersection = ui32(options.MaxConflictFraction * float(objectCount));

        // shortcut for easier sequential one hot bundling
        TVector<TMaybe<ui32>> flatFeatureIdxToBundleIdx(featuresLayout.GetExternalFeatureCount());

        TVector<TExclusiveFeaturesBundle> bundles;
        TVector<TExclusiveFeatureBundleForMerging> bundlesForMerging;

        TFastRng64 rng(0);

        for (auto flatFeatureIdx : flatFeatureIndicesToCalc) {
            const auto featureMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo()[flatFeatureIdx];

            const auto featureType = featuresLayout.GetExternalFeatureType(flatFeatureIdx);
            const auto perTypeFeatureIdx = featuresLayout.GetInternalFeatureIdx(flatFeatureIdx);

            const ui32 featureBinCount
                = (featureType == EFeatureType::Float) ?
                    quantizedFeaturesInfo.GetBinCount(TFloatFeatureIdx(perTypeFeatureIdx)) :
                    quantizedFeaturesInfo.GetUniqueValuesCounts(TCatFeatureIdx(perTypeFeatureIdx)).OnAll;

            // because 0 bin is common for all features in the bundle
            const ui32 binCountInBundleNeeded = featureBinCount - 1;
            if (options.OnlyOneHotsAndBinaryFloats) {
                if (featureBinCount > 2) {
                    continue;
                }
            }
            if (binCountInBundleNeeded >= options.MaxBuckets) {
                continue;
            }

            TConstArrayRef<std::pair<ui32, ui64>> featureNonDefaultMasks
                = featuresNonDefaultMasks[flatFeatureIdx];
            auto featureNonDefaultCount = featuresNonDefaultCounts[flatFeatureIdx];

            auto isCandidateBundle = [&] (ui32 bundleIdx) -> bool {
                auto& bundle = bundles[bundleIdx];

                if (bundle.GetUsedByPartsBinCount() + binCountInBundleNeeded >= options.MaxBuckets) {
                    return false;
                }

                auto& bundleForMerging = bundlesForMerging[bundleIdx];
                auto maxRemaininingIntersectionCount
                    = maxObjectIntersection - bundleForMerging.IntersectionCount;
                if ((bundleForMerging.NonDefaultCount + featuresNonDefaultCounts[flatFeatureIdx])
                     > (objectCount + maxRemaininingIntersectionCount))
                {
                    return false;
                }
                return true;
            };

            auto tryAddToBundle = [&] (ui32 bundleIdx) -> bool {
                auto& bundle = bundles[bundleIdx];
                auto& bundleForMerging = bundlesForMerging[bundleIdx];

                auto maxRemaininingIntersectionCount
                    = maxObjectIntersection - bundleForMerging.IntersectionCount;

                ui32 intersectionCount = CalcIntersectionCount(
                    bundleForMerging.UsedObjects,
                    featureNonDefaultMasks,
                    maxRemaininingIntersectionCount);

                if (intersectionCount <= maxRemaininingIntersectionCount) {
                     AddFeatureToBundle(
                        quantizedFeaturesInfo,
                        flatFeatureIdx,
                        featureNonDefaultMasks,
                        featureNonDefaultCount,
                        binCountInBundleNeeded,
                        intersectionCount,
                        &bundle,
                        &bundleForMerging
                    );
                    flatFeatureIdxToBundleIdx[flatFeatureIdx] = bundleIdx;
                    return true;
                }
                return false;
            };


            THashSet<ui32> checkedBundlesForNeighbors;

            // try neighboring bundles first
            if ((flatFeatureIdx > 0) && flatFeatureIdxToBundleIdx[flatFeatureIdx - 1].Defined()) {
                auto bundleIdx = *(flatFeatureIdxToBundleIdx[flatFeatureIdx - 1]);
                if (isCandidateBundle(bundleIdx) && tryAddToBundle(bundleIdx)) {
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
                if (isCandidateBundle(bundleIdx) && tryAddToBundle(bundleIdx)) {
                    continue;
                }
                checkedBundlesForNeighbors.insert(bundleIdx);
            }

            TVector<ui32> bundlesToCheck;

            for (auto bundleIdx : xrange(bundles.size())) {
                if (!checkedBundlesForNeighbors.contains(bundleIdx) && isCandidateBundle(bundleIdx)) {
                    bundlesToCheck.push_back(bundleIdx);
                }
            }

            const size_t maxBundlesToCheck = options.MaxBundleCandidates - checkedBundlesForNeighbors.size();

            if (bundlesToCheck.size() > maxBundlesToCheck) {
                for (auto i : xrange(maxBundlesToCheck)) {
                    std::swap(bundlesToCheck[i], bundlesToCheck[rng.Uniform(i, bundlesToCheck.size())]);
                }
                bundlesToCheck.resize(maxBundlesToCheck);
            }

            for (auto bundleIdx : bundlesToCheck) {
                if (tryAddToBundle(bundleIdx)) {
                    break;
                }
            }


            if (!flatFeatureIdxToBundleIdx[flatFeatureIdx]) {
                // no bundle found - add new

                flatFeatureIdxToBundleIdx[flatFeatureIdx] = SafeIntegerCast<ui32>(bundles.size());

                TExclusiveFeaturesBundle bundle;
                TExclusiveFeatureBundleForMerging bundleForMerging(objectCount, localExecutor);

                AddFeatureToBundle(
                    quantizedFeaturesInfo,
                    flatFeatureIdx,
                    featureNonDefaultMasks,
                    featureNonDefaultCount,
                    binCountInBundleNeeded,
                    /*intersectionCount*/ 0,
                    &bundle,
                    &bundleForMerging
                );

                bundles.push_back(std::move(bundle));
                bundlesForMerging.push_back(std::move(bundleForMerging));
            }
        }

        TVector<ui32> bundlesForResult;

        // less than sizeof(TBinaryFeaturesPack) * CHAR_BIT binary features
        TVector<ui32> smallBinaryFeaturesOnlyBundles;

        size_t binaryFeaturesInSmallBundlesCount = 0;

        for (auto i : xrange(bundles.size())) {
            auto partCount = bundles[i].Parts.size();

            // 2 is an empiric relative performance constant
            if (bundles[i].IsBinaryFeaturesOnly()
                && (2 * partCount < (sizeof(TBinaryFeaturesPack) * CHAR_BIT)))
            {
                binaryFeaturesInSmallBundlesCount += partCount;
                smallBinaryFeaturesOnlyBundles.push_back(i);
            } else if (partCount > 1) { // don't save bundles with only one feature
                bundlesForResult.push_back(i);
            }
        }

        if (CeilDiv(binaryFeaturesInSmallBundlesCount, sizeof(TBinaryFeaturesPack) * CHAR_BIT)
            >= 2 * smallBinaryFeaturesOnlyBundles.size()) // 2 is an empiric relative performance constant
        {
            // no reason to repack bundles with binary features to packed binary histograms

            for (auto i : smallBinaryFeaturesOnlyBundles) {
                if (bundles[i].Parts.size() > 1) {
                    bundlesForResult.push_back(i);
                }
            }
        }

        if (bundlesForResult.size() == bundles.size()) {
            return bundles;
        }

        Sort(bundlesForResult);

        TVector<TExclusiveFeaturesBundle> result;
        result.reserve(bundlesForResult.size());

        for (auto bundleIdx : bundlesForResult) {
            result.push_back(std::move(bundles[bundleIdx]));
        }

        return result;
    }

    static size_t CalcHistogramReduction(TConstArrayRef<TExclusiveFeaturesBundle> bundles) {
        size_t bundledFeaturesCount = 0;
        for (const auto& bundle : bundles) {
            bundledFeaturesCount += bundle.Parts.size();
        }
        return bundledFeaturesCount - bundles.size();
    }


    TVector<TExclusiveFeaturesBundle> CreateExclusiveFeatureBundles(
        const TRawObjectsData& rawObjectsData,
        const TIncrementalDenseIndexing& rawObjectsDataIncrementalIndexing,
        const TFeaturesLayout& featuresLayout,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TExclusiveFeaturesBundlingOptions& options,
        NPar::ILocalExecutor* localExecutor
    ) {
        const ui32 objectCount = rawObjectsDataIncrementalIndexing.SrcSubsetIndexing.Size();

        if (objectCount == 0) {
            return {};
        }

        const auto featureCount = featuresLayout.GetExternalFeatureCount();
        const auto featuresMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo();

        bool hasSparseFeatures = false;
        TVector<ui32> featureIndicesToCalc;

        for (auto flatFeatureIdx : xrange(featureCount)) {
            const auto& featureMetaInfo = featuresMetaInfo[flatFeatureIdx];
            if (!featureMetaInfo.IsAvailable ||
                (featureMetaInfo.Type == EFeatureType::Text || featureMetaInfo.Type == EFeatureType::Embedding)) {
                continue;
            }

            featureIndicesToCalc.push_back(flatFeatureIdx);

            const ui32 perTypeFeatureIdx = featuresLayout.GetInternalFeatureIdx(flatFeatureIdx);
            bool isSparse = false;
            if (featureMetaInfo.Type == EFeatureType::Float) {
                isSparse = rawObjectsData.FloatFeatures[perTypeFeatureIdx]->IsSparse();
            } else if (featureMetaInfo.Type == EFeatureType::Categorical) {
                isSparse = rawObjectsData.CatFeatures[perTypeFeatureIdx]->IsSparse();
            } else {
                CB_ENSURE(false, featureMetaInfo.Type << " is not supported for feature bundles");
            }

            if (isSparse) {
                hasSparseFeatures = true;
            }
        }

        TFeaturesArraySubsetInvertedIndexing invertedIncrementalIndexing(TFullSubset<ui32>(0));

        if (hasSparseFeatures) {
            invertedIncrementalIndexing = GetInvertedIndexing(
                rawObjectsDataIncrementalIndexing.DstIndexing,
                objectCount,
                localExecutor
            );
        }

        TVector<TVector<std::pair<ui32, ui64>>> featuresNonDefaultMasks(featureCount);
        TVector<ui32> featuresNonDefaultCount(featureCount);

        localExecutor->ExecRange(
            [&] (int featureIdxToCalc) {
                const ui32 flatFeatureIdx = featureIndicesToCalc[featureIdxToCalc];
                const ui32 perTypeFeatureIdx = featuresLayout.GetInternalFeatureIdx(flatFeatureIdx);

                auto& dstMasks = featuresNonDefaultMasks[flatFeatureIdx];
                auto& nonDefaultCount = featuresNonDefaultCount[flatFeatureIdx];

                auto getQuantizedNonDefaultValuesMasks = [&] (const auto& column) {
                    GetQuantizedNonDefaultValuesMasks(
                        column,
                        quantizedFeaturesInfo,
                        rawObjectsDataIncrementalIndexing.SrcSubsetIndexing,
                        invertedIncrementalIndexing,
                        &dstMasks,
                        &nonDefaultCount
                    );
                };

                if (featuresMetaInfo[flatFeatureIdx].Type == EFeatureType::Float) {
                    getQuantizedNonDefaultValuesMasks(*rawObjectsData.FloatFeatures[perTypeFeatureIdx]);
                } else {
                    Y_ASSERT(featuresMetaInfo[flatFeatureIdx].Type == EFeatureType::Categorical);
                    getQuantizedNonDefaultValuesMasks(*rawObjectsData.CatFeatures[perTypeFeatureIdx]);
                }
            },
            0,
            SafeIntegerCast<int>(featureIndicesToCalc.size()),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );

        TVector<ui32> featureIndicesToCalcByNonDefaultCount = featureIndicesToCalc;

        StableSort(
            featureIndicesToCalcByNonDefaultCount,
            [&] (ui32 featureIdx1, ui32 featureIdx2) {
                return featuresNonDefaultCount[featureIdx1] > featuresNonDefaultCount[featureIdx2];
            }
        );

        TVector<TExclusiveFeaturesBundle> results[2];

        TVector<std::function<void()>> tasks;

        auto addTask = [&] (const auto* featureIndices, auto* result) {
            tasks.push_back(
                [&, featureIndices, result] () {
                    *result = CreateExclusiveFeatureBundlesImpl(
                        objectCount,
                        quantizedFeaturesInfo,
                        featuresNonDefaultMasks,
                        featuresNonDefaultCount,
                        *featureIndices,
                        options,
                        localExecutor
                    );
                }
            );
        };

        addTask(&featureIndicesToCalc, &(results[0]));
        addTask(&featureIndicesToCalcByNonDefaultCount, &(results[1]));

        ExecuteTasksInParallel(&tasks, localExecutor);

        if (CalcHistogramReduction(results[1]) > CalcHistogramReduction(results[0])) {
            results[0] = std::move(results[1]);
        }

        return results[0];
    }
}
