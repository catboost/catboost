#include "index_hash_calcer.h"

#include "projection.h"

#include <util/generic/utility.h>
#include <util/generic/xrange.h>


using namespace NCB;


void CalcHashes(
    const TProjection& proj,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    const TPerfectHashedToHashedCatValuesMap* perfectHashedToHashedCatValuesMap,
    bool processAggregatedFeatures,
    ui64* begin,
    ui64* end,
    NPar::TLocalExecutor* localExecutor) {

    const size_t sampleCount = end - begin;
    Y_VERIFY((size_t)featuresSubsetIndexing.Size() == sampleCount);
    if (sampleCount == 0) {
        return;
    }

    ui64* hashArr = begin;


    // [bundleIdx]
    TVector<TVector<TCalcHashInBundleContext>> featuresInBundles(
        objectsDataProvider.GetExclusiveFeatureBundlesSize()
    );

    // [groupIdx]
    TVector<TVector<TCalcHashInGroupContext>> featuresInGroups(
        objectsDataProvider.GetFeaturesGroupsSize()
    );

    // TBinaryFeaturesPack here is actually bit mask to what binary feature in pack are used in projection
    TVector<TBinaryFeaturesPack> binaryFeaturesBitMasks(
        objectsDataProvider.GetBinaryFeaturesPacksSize(),
        TBinaryFeaturesPack(0));

    TVector<TBinaryFeaturesPack> projBinaryFeatureValues(
        objectsDataProvider.GetBinaryFeaturesPacksSize(),
        TBinaryFeaturesPack(0));

    TVector<TCalcHashParams> calcHashParams;

    for (const int featureIdx : proj.CatFeatures) {
        auto catFeatureIdx = TCatFeatureIdx((ui32)featureIdx);
        const auto bundleIdx = objectsDataProvider.GetCatFeatureToExclusiveBundleIndex(catFeatureIdx);
        const auto packIdx = objectsDataProvider.GetCatFeatureToPackedBinaryIndex(catFeatureIdx);
        const auto groupIdx = objectsDataProvider.GetCatFeatureToFeaturesGroupIndex(catFeatureIdx);
        if ((!bundleIdx && !packIdx && !groupIdx) || !processAggregatedFeatures || perfectHashedToHashedCatValuesMap) {
            auto featureCalcHashParams = ExtractColumnLocation<ui32, EFeatureValuesType::PerfectHashedCategorical>(
                bundleIdx,
                packIdx,
                groupIdx,
                [&]() { return *objectsDataProvider.GetCatFeature(*catFeatureIdx); },
                [&](ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeatureBundlesMetaData()[bundleIdx]; },
                [&](ui32 bundleIdx) { return &objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
                [&](ui32 packIdx) { return &objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                [&](ui32 groupIdx) { return &objectsDataProvider.GetFeaturesGroup(groupIdx); }
            );
            if (perfectHashedToHashedCatValuesMap) {
                const auto ohv = MakeArrayRef((*perfectHashedToHashedCatValuesMap)[featureIdx]);
                featureCalcHashParams.CatValuesDecoder = ohv;
            }
            calcHashParams.push_back(featureCalcHashParams);
        } else {
            ExtractIndicesAndMasks(
                bundleIdx,
                packIdx,
                groupIdx,
                /*isBinaryFeatureEquals1*/ true,
                featuresInBundles,
                binaryFeaturesBitMasks,
                projBinaryFeatureValues,
                featuresInGroups,
                [hashArr] (ui32 i, ui32 featureValue) {
                    hashArr[i] = CalcHash(hashArr[i], (ui64)featureValue + 1);
                }
            );
        }
    }

    for (const TBinFeature& feature : proj.BinFeatures) {
        auto floatFeatureIdx = TFloatFeatureIdx((ui32)feature.FloatFeature);
        const auto bundleIdx = objectsDataProvider.GetFloatFeatureToExclusiveBundleIndex(floatFeatureIdx);
        const auto packIdx = objectsDataProvider.GetFloatFeatureToPackedBinaryIndex(floatFeatureIdx);
        const auto groupIdx = objectsDataProvider.GetFloatFeatureToFeaturesGroupIndex(floatFeatureIdx);
        if ((!bundleIdx && !packIdx && !groupIdx) || !processAggregatedFeatures) {
            auto featureCalcHashParams = ExtractColumnLocation<ui8, EFeatureValuesType::QuantizedFloat>(
                bundleIdx,
                packIdx,
                groupIdx,
                [&]() { return *objectsDataProvider.GetFloatFeature(*floatFeatureIdx); },
                [&](ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeatureBundlesMetaData()[bundleIdx]; },
                [&](ui32 bundleIdx) { return &objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
                [&](ui32 packIdx) { return &objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                [&](ui32 groupIdx) { return &objectsDataProvider.GetFeaturesGroup(groupIdx); }
            );
            featureCalcHashParams.SplitIdx = feature.SplitIdx;
            calcHashParams.push_back(featureCalcHashParams);
        } else {
            ExtractIndicesAndMasks(
                bundleIdx,
                packIdx,
                groupIdx,
                /*isBinaryFeatureEquals1*/ true,
                featuresInBundles,
                binaryFeaturesBitMasks,
                projBinaryFeatureValues,
                featuresInGroups,
                [feature, hashArr] (ui32 i, ui32 featureValue) {
                    const bool isTrueFeature = IsTrueHistogram((ui16)featureValue, (ui16)feature.SplitIdx);
                    hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
                }
            );
        }
    }

    const auto& quantizedFeaturesInfo = *objectsDataProvider.GetQuantizedFeaturesInfo();

    for (const TOneHotSplit& feature : proj.OneHotFeatures) {
        auto catFeatureIdx = TCatFeatureIdx((ui32)feature.CatFeatureIdx);

        auto maybeBinaryIndex = objectsDataProvider.GetCatFeatureToPackedBinaryIndex(catFeatureIdx);
        ui32 maxBin = 2;
        if (!maybeBinaryIndex) {
            const auto uniqueValuesCounts = quantizedFeaturesInfo.GetUniqueValuesCounts(
                catFeatureIdx
            );
            maxBin = uniqueValuesCounts.OnLearnOnly;
        }
        const auto bundleIdx = objectsDataProvider.GetCatFeatureToExclusiveBundleIndex(catFeatureIdx);
        const auto groupIdx = objectsDataProvider.GetCatFeatureToFeaturesGroupIndex(catFeatureIdx);
        if ((!bundleIdx && !maybeBinaryIndex) || !processAggregatedFeatures) {
            auto featureCalcHashParams = ExtractColumnLocation<ui32, EFeatureValuesType::PerfectHashedCategorical>(
                bundleIdx,
                maybeBinaryIndex,
                groupIdx,
                [&]() { return *objectsDataProvider.GetCatFeature(*catFeatureIdx); },
                [&](ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeatureBundlesMetaData()[bundleIdx]; },
                [&](ui32 bundleIdx) { return &objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
                [&](ui32 packIdx) { return &objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                [&](ui32 groupIdx) { return &objectsDataProvider.GetFeaturesGroup(groupIdx); }
            );
            featureCalcHashParams.MaxBinAndValue = std::array<ui32, 2>{maxBin, (ui32)feature.Value};
            calcHashParams.push_back(featureCalcHashParams);
        } else {
            ExtractIndicesAndMasks(
                bundleIdx,
                maybeBinaryIndex,
                groupIdx,
                /*isBinaryFeatureEquals1*/ feature.Value == 1,
                featuresInBundles,
                binaryFeaturesBitMasks,
                projBinaryFeatureValues,
                featuresInGroups,
                [feature, hashArr, maxBin] (ui32 i, ui32 featureValue) {
                    const bool isTrueFeature = IsTrueOneHotFeature(Min(featureValue, maxBin), (ui32)feature.Value);
                    hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
                }
            );
        }
    }

    if (!calcHashParams.empty()) {
        featuresSubsetIndexing.ParallelForEachBlockwise(
            [&] (ui32 srcBegin, ui32 srcEnd, ui32 dstBegin, const auto* srcIndices) {
                constexpr ui32 MaxUnrollCount = 16;
                std::array<ui64, MaxUnrollCount> hashes;
                std::array<ui64, MaxUnrollCount> values;
                for (ui32 srcIdx = srcBegin, dstIdx = dstBegin; srcIdx < srcEnd; srcIdx += MaxUnrollCount, dstIdx += MaxUnrollCount) {
                    const ui32 unrollCount = Min(MaxUnrollCount, srcEnd - srcIdx);
                    hashes.fill(0);
                    for (const auto& featureCalcHashParams : calcHashParams) {
                        if (srcIndices) {
                            featureCalcHashParams.GatherValues(srcIdx, unrollCount, srcIndices, values);
                        } else {
                            featureCalcHashParams.GatherValues(srcIdx, unrollCount, values);
                        }

                        if (featureCalcHashParams.CatValuesDecoder) {
                            const auto valuesDecoder = featureCalcHashParams.CatValuesDecoder.GetRef();
                            for (ui32 unrollIdx : xrange(unrollCount)) {
                                hashes[unrollIdx] = CalcHash(hashes[unrollIdx], (int)valuesDecoder[values[unrollIdx]]);
                            }
                        } else if (featureCalcHashParams.SplitIdx) {
                            const auto splitIdx = featureCalcHashParams.SplitIdx.GetRef();
                            for (ui32 unrollIdx : xrange(unrollCount)) {
                                const auto isTrueFeature = IsTrueHistogram<ui16>(values[unrollIdx], splitIdx);
                                hashes[unrollIdx] = CalcHash(hashes[unrollIdx], isTrueFeature);
                            }
                        } else if (featureCalcHashParams.MaxBinAndValue) {
                            const auto maxBinAndValue = featureCalcHashParams.MaxBinAndValue.GetRef();
                            for (ui32 unrollIdx : xrange(unrollCount)) {
                                const auto isTrueFeature = IsTrueOneHotFeature(Min<ui32>(values[unrollIdx], maxBinAndValue[0]), (ui32)maxBinAndValue[1]);
                                hashes[unrollIdx] = CalcHash(hashes[unrollIdx], isTrueFeature);
                            }
                        } else {
                            for (ui32 unrollIdx : xrange(unrollCount)) {
                                hashes[unrollIdx] = CalcHash(hashes[unrollIdx], values[unrollIdx] + 1);
                            }
                        }
                    }
                    Copy(hashes.begin(), hashes.begin() + unrollCount, hashArr + dstIdx);
                }
            },
            localExecutor
        );
    }

    if (processAggregatedFeatures) {
        for (size_t bundleIdx : xrange(featuresInBundles.size())) {
            TConstArrayRef<TCalcHashInBundleContext> featuresInBundle = featuresInBundles[bundleIdx];
            if (featuresInBundle.empty()) {
                continue;
            }

            const auto& metaData = objectsDataProvider.GetExclusiveFeatureBundlesMetaData()[bundleIdx];

            TVector<TBoundsInBundle> selectedBounds;

            for (const auto& featureInBundle : featuresInBundle) {
                selectedBounds.push_back(metaData.Parts[featureInBundle.InBundleIdx].Bounds);
            }

            auto processBundleValue = [&] (ui32 i, ui16 bundleValue) {
                for (auto selectedFeatureIdx : xrange(featuresInBundle.size())) {
                    featuresInBundle[selectedFeatureIdx].CalcHashCallback(
                        i,
                        GetBinFromBundle<decltype(bundleValue)>(
                            bundleValue,
                            selectedBounds[selectedFeatureIdx]
                        )
                    );
                }
            };

            ProcessColumnForCalcHashes(
                objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx),
                featuresSubsetIndexing,
                std::move(processBundleValue),
                localExecutor
            );
        }

        for (size_t packIdx : xrange(binaryFeaturesBitMasks.size())) {
            TBinaryFeaturesPack bitMask = binaryFeaturesBitMasks[packIdx];
            if (!bitMask) {
                continue;
            }

            TBinaryFeaturesPack packProjBinaryFeatureValues = projBinaryFeatureValues[packIdx];

            auto getBinFromHistogramValue = [=] (TBinaryFeaturesPack defaultPackValue) -> ui64 {
                // returns default 'b' for CalcHash
                return (ui64)((~(defaultPackValue ^ packProjBinaryFeatureValues)) & bitMask) + (ui64)bitMask;
            };

            ProcessColumnForCalcHashes(
                objectsDataProvider.GetBinaryFeaturesPack(packIdx),
                featuresSubsetIndexing,
                std::move(getBinFromHistogramValue),
                [=] (ui32 i, ui64 b) {
                    hashArr[i] = CalcHash(hashArr[i], b);
                },
                localExecutor
            );
        }

        for (size_t groupIdx : xrange(featuresInGroups.size())) {
            TConstArrayRef<TCalcHashInGroupContext> featuresInGroup = featuresInGroups[groupIdx];
            if (featuresInGroup.empty()) {
                continue;
            }

            auto processGroupValue = [&] (ui32 i, auto groupValue) {
                for (const auto& selectedFeature : featuresInGroup) {
                    selectedFeature.CalcHashCallback(
                        i,
                        GetPartValueFromGroup(groupValue, selectedFeature.InGroupIdx)
                    );
                }
            };

            ProcessColumnForCalcHashes(
                objectsDataProvider.GetFeaturesGroup(groupIdx),
                featuresSubsetIndexing,
                std::move(processGroupValue),
                localExecutor
            );
        }
    }
}


/// Compute reindexHash and reindex hash values in range [begin,end).
size_t ComputeReindexHash(ui64 topSize, TDenseHash<ui64, ui32>* reindexHashPtr, ui64* begin, ui64* end) {
    auto& reindexHash = *reindexHashPtr;
    auto* hashArr = begin;
    size_t learnSize = end - begin;
    ui32 counter = 0;
    if (topSize > learnSize) {
        for (size_t i = 0; i < learnSize; ++i) {
            auto p = reindexHash.emplace(hashArr[i], counter);
            if (p.second) {
                ++counter;
            }
            hashArr[i] = p.first->second;
        }
    } else {
        for (size_t i = 0; i < learnSize; ++i) {
            ++reindexHash[hashArr[i]];
        }

        if (reindexHash.Size() <= topSize) {
            for (auto& it : reindexHash) {
                it.second = counter;
                ++counter;
            }
            for (size_t i = 0; i < learnSize; ++i) {
                hashArr[i] = reindexHash.Value(hashArr[i], 0);
            }
        } else {
            // Limit reindexHash to topSize buckets
            using TFreqPair = std::pair<ui64, ui32>;
            TVector<TFreqPair> freqValList;

            freqValList.reserve(reindexHash.Size());
            for (const auto& it : reindexHash) {
                freqValList.emplace_back(it.first, it.second);
            }
            std::nth_element(
                freqValList.begin(),
                freqValList.begin() + topSize,
                freqValList.end(),
                [](const TFreqPair& a, const TFreqPair& b) {
                    return a.second > b.second;
                });

            reindexHash.MakeEmpty();
            for (ui32 i = 0; i < topSize; ++i) {
                reindexHash[freqValList[i].first] = i;
            }
            for (ui64* hash = begin; hash != end; ++hash) {
               if (auto* p = reindexHash.FindPtr(*hash)) {
                   *hash = *p;
               } else {
                   *hash = reindexHash.Size() - 1;
               }
            }
        }
    }
    return reindexHash.Size();
}

/// Update reindexHash and reindex hash values in range [begin,end).
size_t UpdateReindexHash(TDenseHash<ui64, ui32>* reindexHashPtr, ui64* begin, ui64* end) {
    auto& reindexHash = *reindexHashPtr;
    ui32 counter = reindexHash.Size();
    for (ui64* hash = begin; hash != end; ++hash) {
        auto p = reindexHash.emplace(*hash, counter);
        if (p.second) {
            *hash = counter++;
        } else {
            *hash = p.first->second;
        }
    }
    return reindexHash.Size();
}
