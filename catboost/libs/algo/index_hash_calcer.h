#pragma once

#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/helpers/clear_array.h>

#include <library/containers/dense_hash/dense_hash.h>

#include <util/generic/vector.h>
#include <util/system/yassert.h>

#include <functional>


struct TProjection;


struct TCalcHashInBundleContext {
    ui32 InBundleIdx = 0;
    std::function<void(ui32, ui32)> CalcHashCallback;
};


template <class T, NCB::EFeatureValuesType FeatureValuesType, class F> // F args are (index, value)
inline void ProcessColumnForCalcHashes(
    const NCB::TTypedFeatureValuesHolder<T, FeatureValuesType>& column,
    const NCB::TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    F&& f) {

    using TDenseHolder = NCB::TCompressedValuesHolderImpl<T, FeatureValuesType>;

    if (const auto* denseColumnData = dynamic_cast<const TDenseHolder*>(&column)) {
        const TCompressedArray& compressedArray = *denseColumnData->GetCompressedData().GetSrc();

        NCB::DispatchBitsPerKeyToDataType(
            compressedArray,
            "ProcessColumnForCalcHashes",
            [&] (const auto* histogram) {
                featuresSubsetIndexing.ForEach(
                    [histogram, f] (ui32 i, ui32 srcIdx) {
                        f(i, histogram[srcIdx]);
                    }
                );
            }
        );
    } else {
        Y_FAIL("ProcessColumnForCalcHashes: unexpected column type");
    }
}

template <class T, NCB::EFeatureValuesType FeatureValuesType, class TGetBinFromHistogramValue, class F>
inline void ProcessColumnForCalcHashes(
    const NCB::TTypedFeatureValuesHolder<T, FeatureValuesType>& column,
    const NCB::TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    TGetBinFromHistogramValue&& getBinFromHistogramValue,
    F&& f) {

    ProcessColumnForCalcHashes(
        column,
        featuresSubsetIndexing,
        [f, getBinFromHistogramValue] (ui32 i, auto value) {
            f(i, getBinFromHistogramValue(value));
        });
}

template <class T, NCB::EFeatureValuesType FeatureValuesType, class F>
inline void ProcessFeatureForCalcHashes(
    TMaybe<NCB::TExclusiveBundleIndex> maybeExclusiveBundleIndex,
    TMaybe<NCB::TPackedBinaryIndex> maybeBinaryIndex,
    const NCB::TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    bool processBundledAndBinaryFeaturesInPacks,
    bool isBinaryFeatureEquals1, // used only if processBinary
    TArrayRef<TVector<TCalcHashInBundleContext>> featuresInBundles,
    TArrayRef<NCB::TBinaryFeaturesPack> binaryFeaturesBitMasks,
    TArrayRef<NCB::TBinaryFeaturesPack> projBinaryFeatureValues,
    std::function<const NCB::TTypedFeatureValuesHolder<T, FeatureValuesType>*()>&& getFeatureColumn,
    std::function<const NCB::TExclusiveFeaturesBundle(ui32)>&& getExclusiveFeatureBundleMetaData,
    std::function<const NCB::TExclusiveFeatureBundleHolder*(ui32)>&& getExclusiveFeatureBundle,
    std::function<const NCB::TBinaryPacksHolder*(ui32)>&& getBinaryFeaturesPack,
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
            const ui32 bundleIdx = maybeExclusiveBundleIndex->BundleIdx;
            const auto& metaData = getExclusiveFeatureBundleMetaData(bundleIdx);
            const NCB::TBoundsInBundle boundsInBundle
                = metaData.Parts[maybeExclusiveBundleIndex->InBundleIdx].Bounds;

            ProcessColumnForCalcHashes(
                *getExclusiveFeatureBundle(bundleIdx),
                featuresSubsetIndexing,
                [boundsInBundle] (auto bundleData) {
                    return NCB::GetBinFromBundle<decltype(bundleData)>(bundleData, boundsInBundle);
                },
                std::move(f)
            );
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

            ProcessColumnForCalcHashes(
                *getBinaryFeaturesPack(maybeBinaryIndex->PackIdx),
                featuresSubsetIndexing,
                [bitMask, bitIdx] (NCB::TBinaryFeaturesPack featuresPack) {
                    return (featuresPack & bitMask) >> bitIdx;
                },
                std::move(f)
            );
        }
    } else {
        ProcessColumnForCalcHashes(
            *getFeatureColumn(),
            featuresSubsetIndexing,
            [] (auto value) { return value; },
            std::move(f)
        );
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
void CalcHashes(
    const TProjection& proj,
    const NCB::TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const NCB::TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    const NCB::TPerfectHashedToHashedCatValuesMap* perfectHashedToHashedCatValuesMap,
    bool processBundledAndBinaryFeaturesInPacks,
    ui64* begin,
    ui64* end);


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
