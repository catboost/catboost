#pragma once

#include "projection.h"

#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/helpers/clear_array.h>

#include <library/containers/dense_hash/dense_hash.h>

#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>

#include <functional>


struct TCalcHashInBundleContext {
    ui32 InBundleIdx = 0;
    std::function<void(ui32, ui32)> CalcHashCallback;
};


template <class IFeatureColumn, class F>
inline void ProcessFeatureForCalcHashes(
    TMaybe<NCB::TExclusiveBundleIndex> maybeExclusiveBundleIndex,
    TMaybe<NCB::TPackedBinaryIndex> maybeBinaryIndex,
    const NCB::TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    bool processBundledAndBinaryFeaturesInPacks,
    bool isBinaryFeatureEquals1, // used only if processBinary
    TArrayRef<TVector<TCalcHashInBundleContext>> featuresInBundles,
    TArrayRef<NCB::TBinaryFeaturesPack> binaryFeaturesBitMasks,
    TArrayRef<NCB::TBinaryFeaturesPack> projBinaryFeatureValues,
    std::function<const IFeatureColumn*()>&& getFeatureColumn,
    std::function<NCB::TFeaturesBundleArraySubset(int)>&& getExclusiveFeatureBundle,
    std::function<NCB::TPackedBinaryFeaturesArraySubset(int)>&& getBinaryFeaturesPack,
    F&& f) {

    if (maybeExclusiveBundleIndex) {
        if (processBundledAndBinaryFeaturesInPacks) {
            TCalcHashInBundleContext calcHashInBundleContext;
            calcHashInBundleContext.InBundleIdx = maybeExclusiveBundleIndex->InBundleIdx;
            calcHashInBundleContext.CalcHashCallback = f;

            featuresInBundles[maybeExclusiveBundleIndex->BundleIdx].push_back(
                std::move(calcHashInBundleContext)
            );
        } else {
            NCB::TFeaturesBundleArraySubset featuresBundleArraySubset = getExclusiveFeatureBundle(
                maybeExclusiveBundleIndex->BundleIdx
            );
            const auto& metaData = *featuresBundleArraySubset.MetaData;
            auto boundsInBundle = metaData.Parts[maybeExclusiveBundleIndex->InBundleIdx].Bounds;

            auto iterateFunction = [&] (const auto* bundlesSrcData) {
                featuresSubsetIndexing.ForEach(
                    [&, bundlesSrcData, boundsInBundle] (ui32 i, ui32 srcIdx) {
                        auto bundleData = bundlesSrcData[srcIdx];
                        f(i, NCB::GetBinFromBundle<decltype(bundleData)>(bundleData, boundsInBundle));
                    }
                );
            };

            switch (metaData.SizeInBytes) {
                case 1:
                    iterateFunction(featuresBundleArraySubset.SrcData.data());
                    break;
                case 2:
                    iterateFunction((const ui16*)featuresBundleArraySubset.SrcData.data());
                    break;
                default:
                    CB_ENSURE_INTERNAL(
                        false,
                        "unsupported Bundle SizeInBytes = " << metaData.SizeInBytes
                    );
            }
        }
    } else if (maybeBinaryIndex) {
        NCB::TBinaryFeaturesPack bitMask = NCB::TBinaryFeaturesPack(1) << maybeBinaryIndex->BitIdx;
        if (processBundledAndBinaryFeaturesInPacks) {
            binaryFeaturesBitMasks[maybeBinaryIndex->PackIdx] |= bitMask;

            if (isBinaryFeatureEquals1) {
                projBinaryFeatureValues[maybeBinaryIndex->PackIdx] |= bitMask;
            }
        } else {
            NCB::TBinaryFeaturesPack bitIdx = maybeBinaryIndex->BitIdx;
            NCB::TPackedBinaryFeaturesArraySubset packSubset = getBinaryFeaturesPack(maybeBinaryIndex->PackIdx);

            NCB::TPackedBinaryFeaturesArraySubset(
                packSubset.GetSrc(),
                &featuresSubsetIndexing
            ).ForEach(
                [bitMask, bitIdx, f = std::move(f)] (ui32 i, NCB::TBinaryFeaturesPack featuresPack) {
                    f(i, (featuresPack & bitMask) >> bitIdx);
                }
            );
        }
    } else {
        dynamic_cast<const NCB::TCompressedValuesHolderImpl<IFeatureColumn>*>(getFeatureColumn())
            ->ForEach(std::move(f), &featuresSubsetIndexing);
    }
}


/// Calculate document hashes into range [begin,end) for CTR bucket identification.
/// @param proj - Projection delivering the feature ids to hash
/// @param objectsDataProvider - Values of features to hash
/// @param featuresSubsetIndexing - Use these indices when accessing raw arrays data
/// @param perfectHashedToHashedCatValuesMap - if not nullptr use it to Hash original hashed cat values
//                                             if nullptr - used perfectHashed values
/// @param processBundledAndBinaryFeaturesInPacks - process bundled and binary features in packs.
///                                       Faster, but not compatible with current model format.
///                                       So, enabled only during training, disabled for FinalCtr.
/// @param begin, @param end - Result range
inline void CalcHashes(
    const TProjection& proj,
    const NCB::TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const NCB::TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    const NCB::TPerfectHashedToHashedCatValuesMap* perfectHashedToHashedCatValuesMap,
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
    TVector<NCB::TBinaryFeaturesPack> binaryFeaturesBitMasks(
        objectsDataProvider.GetBinaryFeaturesPacksSize(),
        NCB::TBinaryFeaturesPack(0));

    TVector<NCB::TBinaryFeaturesPack> projBinaryFeatureValues(
        objectsDataProvider.GetBinaryFeaturesPacksSize(),
        NCB::TBinaryFeaturesPack(0));

    if (perfectHashedToHashedCatValuesMap) {
        for (const int featureIdx : proj.CatFeatures) {
            auto catFeatureIdx = NCB::TCatFeatureIdx((ui32)featureIdx);

            const auto& ohv = (*perfectHashedToHashedCatValuesMap)[featureIdx];

            ProcessFeatureForCalcHashes<NCB::IQuantizedCatValuesHolder>(
                objectsDataProvider.GetCatFeatureToExclusiveBundleIndex(catFeatureIdx),
                objectsDataProvider.GetCatFeatureToPackedBinaryIndex(catFeatureIdx),
                featuresSubsetIndexing,
                /*processBundledAndBinaryFeaturesInPacks*/ false,
                /*isBinaryFeatureEquals1*/ false, // unused
                TArrayRef<TVector<TCalcHashInBundleContext>>(), // unused
                TArrayRef<NCB::TBinaryFeaturesPack>(), // unused
                TArrayRef<NCB::TBinaryFeaturesPack>(), // unused
                [&]() { return *objectsDataProvider.GetCatFeature(*catFeatureIdx); },
                [&](ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
                [&](ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                [hashArr, &ohv] (ui32 i, ui32 featureValue) {
                    hashArr[i] = CalcHash(hashArr[i], (ui64)(int)ohv[featureValue]);
                }
            );
        }
    } else {
        for (const int featureIdx : proj.CatFeatures) {
            auto catFeatureIdx = NCB::TCatFeatureIdx((ui32)featureIdx);
            ProcessFeatureForCalcHashes<NCB::IQuantizedCatValuesHolder>(
                objectsDataProvider.GetCatFeatureToExclusiveBundleIndex(catFeatureIdx),
                objectsDataProvider.GetCatFeatureToPackedBinaryIndex(catFeatureIdx),
                featuresSubsetIndexing,
                processBundledAndBinaryFeaturesInPacks,
                /*isBinaryFeatureEquals1*/ true,
                featuresInBundles,
                binaryFeaturesBitMasks,
                projBinaryFeatureValues,
                [&]() { return *objectsDataProvider.GetCatFeature(*catFeatureIdx); },
                [&](ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
                [&](ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                [hashArr] (ui32 i, ui32 featureValue) {
                    hashArr[i] = CalcHash(hashArr[i], (ui64)featureValue + 1);
                }
            );
        }
    }


    for (const TBinFeature& feature : proj.BinFeatures) {
        auto floatFeatureIdx = NCB::TFloatFeatureIdx((ui32)feature.FloatFeature);
        ProcessFeatureForCalcHashes<NCB::IQuantizedFloatValuesHolder>(
            objectsDataProvider.GetFloatFeatureToExclusiveBundleIndex(floatFeatureIdx),
            objectsDataProvider.GetFloatFeatureToPackedBinaryIndex(floatFeatureIdx),
            featuresSubsetIndexing,
            processBundledAndBinaryFeaturesInPacks,
            /*isBinaryFeatureEquals1*/ 1,
            featuresInBundles,
            binaryFeaturesBitMasks,
            projBinaryFeatureValues,
            [&]() { return *objectsDataProvider.GetFloatFeature(*floatFeatureIdx); },
            [&](ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
            [&](ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
            [feature, hashArr] (ui32 i, ui32 featureValue) {
                const bool isTrueFeature = IsTrueHistogram((ui16)featureValue, (ui16)feature.SplitIdx);
                hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
            }
        );
    }

    const auto& quantizedFeaturesInfo = *objectsDataProvider.GetQuantizedFeaturesInfo();

    for (const TOneHotSplit& feature : proj.OneHotFeatures) {
        auto catFeatureIdx = NCB::TCatFeatureIdx((ui32)feature.CatFeatureIdx);

        auto maybeBinaryIndex = objectsDataProvider.GetCatFeatureToPackedBinaryIndex(catFeatureIdx);
        ui32 maxBin = 2;
        if (!maybeBinaryIndex) {
            const auto uniqueValuesCounts = quantizedFeaturesInfo.GetUniqueValuesCounts(
                catFeatureIdx
            );
            maxBin = uniqueValuesCounts.OnLearnOnly;
        }

        ProcessFeatureForCalcHashes<NCB::IQuantizedCatValuesHolder>(
            objectsDataProvider.GetCatFeatureToExclusiveBundleIndex(catFeatureIdx),
            maybeBinaryIndex,
            featuresSubsetIndexing,
            processBundledAndBinaryFeaturesInPacks,
            /*isBinaryFeatureEquals1*/ feature.Value == 1,
            featuresInBundles,
            binaryFeaturesBitMasks,
            projBinaryFeatureValues,
            [&]() { return *objectsDataProvider.GetCatFeature(*catFeatureIdx); },
            [&](ui32 bundleIdx) { return objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx); },
            [&](ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
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

            NCB::TFeaturesBundleArraySubset featuresBundleArraySubset
                = objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx);

            const auto& metaData = *featuresBundleArraySubset.MetaData;

            TVector<NCB::TBoundsInBundle> selectedBounds;

            for (const auto& featureInBundle : featuresInBundle) {
                selectedBounds.push_back(metaData.Parts[featureInBundle.InBundleIdx].Bounds);
            }

            auto iterateFunction = [&] (const auto* bundlesSrcData) {
                featuresSubsetIndexing.ForEach(
                    [selectedBounds, featuresInBundle, bundlesSrcData] (ui32 i, ui32 srcIdx) {
                        auto bundleData = bundlesSrcData[srcIdx];

                        for (auto selectedFeatureIdx : xrange(featuresInBundle.size())) {
                            featuresInBundle[selectedFeatureIdx].CalcHashCallback(
                                i,
                                NCB::GetBinFromBundle<decltype(bundleData)>(
                                    bundleData,
                                    selectedBounds[selectedFeatureIdx]
                                )
                            );
                        }
                    }
                );
            };

            switch (metaData.SizeInBytes) {
                case 1:
                    iterateFunction(featuresBundleArraySubset.SrcData.data());
                    break;
                case 2:
                    iterateFunction((const ui16*)featuresBundleArraySubset.SrcData.data());
                    break;
                default:
                    CB_ENSURE_INTERNAL(
                        false,
                        "unsupported Bundle SizeInBytes = " << metaData.SizeInBytes
                    );
            }
        }

        for (size_t packIdx : xrange(binaryFeaturesBitMasks.size())) {
            NCB::TBinaryFeaturesPack bitMask = binaryFeaturesBitMasks[packIdx];
            if (!bitMask) {
                continue;
            }
            NCB::TBinaryFeaturesPack packProjBinaryFeatureValues = projBinaryFeatureValues[packIdx];

            NCB::TPackedBinaryFeaturesArraySubset(
                objectsDataProvider.GetBinaryFeaturesPack(packIdx).GetSrc(),
                &featuresSubsetIndexing
            ).ForEach(
                [bitMask, packProjBinaryFeatureValues, hashArr] (
                    ui32 i,
                    NCB::TBinaryFeaturesPack binaryFeaturesPack
                ) {
                    hashArr[i] = CalcHash(
                        hashArr[i],
                        (ui64)((~(binaryFeaturesPack ^ packProjBinaryFeatureValues)) & bitMask) + (ui64)bitMask
                    );
                }
            );
        }
    }
}

/// Compute reindexHash and reindex hash values in range [begin,end).
/// After reindex, hash values belong to [0, reindexHash.Size()].
/// If reindexHash would become larger than topSize, keep only topSize most
/// frequent mappings and map other hash values to value reindexHash.Size().
/// @return the size of reindexHash.
size_t ComputeReindexHash(ui64 topSize, TDenseHash<ui64, ui32>* reindexHashPtr, ui64* begin, ui64* end);

/// Update reindexHash and reindex hash values in range [begin,end).
/// If a hash value is not present in reindexHash, then update reindexHash for that value.
/// @return the size of updated reindexHash.
size_t UpdateReindexHash(TDenseHash<ui64, ui32>* reindexHashPtr, ui64* begin, ui64* end);
