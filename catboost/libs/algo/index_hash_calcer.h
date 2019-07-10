#pragma once

#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/helpers/clear_array.h>

#include <library/containers/dense_hash/dense_hash.h>

#include <util/generic/vector.h>

#include <functional>


struct TProjection;


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
