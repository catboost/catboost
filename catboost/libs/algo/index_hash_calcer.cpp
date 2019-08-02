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
    bool processBundledAndBinaryFeaturesInPacks,
    ui64* begin,
    ui64* end) {

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

    // TBinaryFeaturesPack here is actually bit mask to what binary feature in pack are used in projection
    TVector<TBinaryFeaturesPack> binaryFeaturesBitMasks(
        objectsDataProvider.GetBinaryFeaturesPacksSize(),
        TBinaryFeaturesPack(0));

    TVector<TBinaryFeaturesPack> projBinaryFeatureValues(
        objectsDataProvider.GetBinaryFeaturesPacksSize(),
        TBinaryFeaturesPack(0));

    if (perfectHashedToHashedCatValuesMap) {
        for (const int featureIdx : proj.CatFeatures) {
            auto catFeatureIdx = TCatFeatureIdx((ui32)featureIdx);

            const auto& ohv = (*perfectHashedToHashedCatValuesMap)[featureIdx];

            ProcessFeatureForCalcHashes<ui32, EFeatureValuesType::PerfectHashedCategorical>(
                objectsDataProvider.GetCatFeatureToExclusiveBundleIndex(catFeatureIdx),
                objectsDataProvider.GetCatFeatureToPackedBinaryIndex(catFeatureIdx),
                featuresSubsetIndexing,
                /*processBundledAndBinaryFeaturesInPacks*/ false,
                /*isBinaryFeatureEquals1*/ false, // unused
                TArrayRef<TVector<TCalcHashInBundleContext>>(), // unused
                TArrayRef<TBinaryFeaturesPack>(), // unused
                TArrayRef<TBinaryFeaturesPack>(), // unused
                [&]() { return *objectsDataProvider.GetCatFeature(*catFeatureIdx); },
                [&](ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeatureBundlesMetaData()[bundleIdx]; },
                [&](ui32 bundleIdx) { return &objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
                [&](ui32 packIdx) { return &objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                [hashArr, &ohv] (ui32 i, ui32 featureValue) {
                    hashArr[i] = CalcHash(hashArr[i], (ui64)(int)ohv[featureValue]);
                }
            );
        }
    } else {
        for (const int featureIdx : proj.CatFeatures) {
            auto catFeatureIdx = TCatFeatureIdx((ui32)featureIdx);
            ProcessFeatureForCalcHashes<ui32, EFeatureValuesType::PerfectHashedCategorical>(
                objectsDataProvider.GetCatFeatureToExclusiveBundleIndex(catFeatureIdx),
                objectsDataProvider.GetCatFeatureToPackedBinaryIndex(catFeatureIdx),
                featuresSubsetIndexing,
                processBundledAndBinaryFeaturesInPacks,
                /*isBinaryFeatureEquals1*/ true,
                featuresInBundles,
                binaryFeaturesBitMasks,
                projBinaryFeatureValues,
                [&]() { return *objectsDataProvider.GetCatFeature(*catFeatureIdx); },
                [&](ui32 bundleIdx) {
                    return objectsDataProvider.GetExclusiveFeatureBundlesMetaData()[bundleIdx];
                },
                [&](ui32 bundleIdx) { return &objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
                [&](ui32 packIdx) { return &objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                [hashArr] (ui32 i, ui32 featureValue) {
                    hashArr[i] = CalcHash(hashArr[i], (ui64)featureValue + 1);
                }
            );
        }
    }


    for (const TBinFeature& feature : proj.BinFeatures) {
        auto floatFeatureIdx = TFloatFeatureIdx((ui32)feature.FloatFeature);
        ProcessFeatureForCalcHashes<ui8, EFeatureValuesType::QuantizedFloat>(
            objectsDataProvider.GetFloatFeatureToExclusiveBundleIndex(floatFeatureIdx),
            objectsDataProvider.GetFloatFeatureToPackedBinaryIndex(floatFeatureIdx),
            featuresSubsetIndexing,
            processBundledAndBinaryFeaturesInPacks,
            /*isBinaryFeatureEquals1*/ 1,
            featuresInBundles,
            binaryFeaturesBitMasks,
            projBinaryFeatureValues,
            [&]() { return *objectsDataProvider.GetFloatFeature(*floatFeatureIdx); },
            [&](ui32 bundleIdx) {
                return objectsDataProvider.GetExclusiveFeatureBundlesMetaData()[bundleIdx];
            },
            [&](ui32 bundleIdx) { return &objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
            [&](ui32 packIdx) { return &objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
            [feature, hashArr] (ui32 i, ui32 featureValue) {
                const bool isTrueFeature = IsTrueHistogram((ui16)featureValue, (ui16)feature.SplitIdx);
                hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
            }
        );
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

        ProcessFeatureForCalcHashes<ui32, EFeatureValuesType::PerfectHashedCategorical>(
            objectsDataProvider.GetCatFeatureToExclusiveBundleIndex(catFeatureIdx),
            maybeBinaryIndex,
            featuresSubsetIndexing,
            processBundledAndBinaryFeaturesInPacks,
            /*isBinaryFeatureEquals1*/ feature.Value == 1,
            featuresInBundles,
            binaryFeaturesBitMasks,
            projBinaryFeatureValues,
            [&]() { return *objectsDataProvider.GetCatFeature(*catFeatureIdx); },
            [&](ui32 bundleIdx) {
                return objectsDataProvider.GetExclusiveFeatureBundlesMetaData()[bundleIdx];
            },
            [&](ui32 bundleIdx) { return &objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
            [&](ui32 packIdx) { return &objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
            [feature, hashArr, maxBin] (ui32 i, ui32 featureValue) {
                const bool isTrueFeature = IsTrueOneHotFeature(Min(featureValue, maxBin), (ui32)feature.Value);
                hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
            }
        );
    }

    if (processBundledAndBinaryFeaturesInPacks) {
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
                std::move(processBundleValue)
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
                }
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
