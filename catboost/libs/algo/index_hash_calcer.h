#pragma once

#include "projection.h"

#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/helpers/clear_array.h>

#include <library/containers/dense_hash/dense_hash.h>
#include <library/containers/stack_vector/stack_vec.h>

#include <util/generic/utility.h>
#include <util/generic/xrange.h>

#include <functional>


template <class IFeatureColumn, class F>
inline void ProcessFeatureForCalcHashes(
    TMaybe<NCB::TPackedBinaryIndex> maybeBinaryIndex,
    const NCB::TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    bool processBinaryInPacks,
    bool isBinaryFeatureEquals1, // used only if processBinary
    TArrayRef<NCB::TBinaryFeaturesPack> binaryFeaturesBitMasks,
    TArrayRef<NCB::TBinaryFeaturesPack> projBinaryFeatureValues,
    std::function<const IFeatureColumn*()>&& getFeatureColumn,
    std::function<NCB::TPackedBinaryFeaturesArraySubset(int)>&& getBinaryFeaturesPack,
    F&& f) {

    if (maybeBinaryIndex) {
        NCB::TBinaryFeaturesPack bitMask = NCB::TBinaryFeaturesPack(1) << maybeBinaryIndex->BitIdx;
        if (processBinaryInPacks) {
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
        NCB::TConstPtrArraySubset<typename IFeatureColumn::TValueType>(
            dynamic_cast<const NCB::TCompressedValuesHolderImpl<IFeatureColumn>*>(getFeatureColumn())
                ->GetArrayData().GetSrc(),
            &featuresSubsetIndexing
        ).ForEach(std::move(f));
    }
}


/// Calculate document hashes into range [begin,end) for CTR bucket identification.
/// @param proj - Projection delivering the feature ids to hash
/// @param objectsDataProvider - Values of features to hash
/// @param featuresSubsetIndexing - Use these indices when accessing raw arrays data
/// @param perfectHashedToHashedCatValuesMap - if not nullptr use it to Hash original hashed cat values
//                                             if nullptr - used perfectHashed values
/// @param processBinaryFeaturesInPacks - process binary features in packs. Faster, but not compatible with
///                                       current model format.
///                                       So, enabled only during training, disabled for FinalCtr.
/// @param begin, @param end - Result range
inline void CalcHashes(const TProjection& proj,
                       const NCB::TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
                       const NCB::TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
                       const NCB::TPerfectHashedToHashedCatValuesMap* perfectHashedToHashedCatValuesMap,
                       bool processBinaryFeaturesInPacks,
                       ui64* begin,
                       ui64* end) {
    const size_t sampleCount = end - begin;
    Y_VERIFY((size_t)featuresSubsetIndexing.Size() == sampleCount);
    if (sampleCount == 0) {
        return;
    }

    ui64* hashArr = begin;
    if (perfectHashedToHashedCatValuesMap) {
        for (const int featureIdx : proj.CatFeatures) {
            auto catFeatureIdx = NCB::TCatFeatureIdx((ui32)featureIdx);

            const auto& ohv = (*perfectHashedToHashedCatValuesMap)[featureIdx];

            ProcessFeatureForCalcHashes<NCB::IQuantizedCatValuesHolder>(
                objectsDataProvider.GetCatFeatureToPackedBinaryIndex(catFeatureIdx),
                featuresSubsetIndexing,
                /*processBinaryInPacks*/ false,
                /*isBinaryFeatureEquals1*/ false, // unused
                TArrayRef<NCB::TBinaryFeaturesPack>(), // unused
                TArrayRef<NCB::TBinaryFeaturesPack>(), // unused
                [&]() { return *objectsDataProvider.GetCatFeature(*catFeatureIdx); },
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
                objectsDataProvider.GetCatFeatureToPackedBinaryIndex(catFeatureIdx),
                featuresSubsetIndexing,
                /*processBinaryInPacks*/ false,
                /*isBinaryFeatureEquals1*/ false, // unused
                TArrayRef<NCB::TBinaryFeaturesPack>(), // unused
                TArrayRef<NCB::TBinaryFeaturesPack>(), // unused
                [&]() { return *objectsDataProvider.GetCatFeature(*catFeatureIdx); },
                [&](ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
                [hashArr] (ui32 i, ui32 featureValue) {
                    hashArr[i] = CalcHash(hashArr[i], (ui64)featureValue + 1);
                }
            );
        }
    }

    // TBinaryFeaturesPack here is actually bit mask to what binary feature in pack are used in projection
    TStackVec<NCB::TBinaryFeaturesPack> binaryFeaturesBitMasks(
        objectsDataProvider.GetBinaryFeaturesPacksSize(),
        NCB::TBinaryFeaturesPack(0));

    TStackVec<NCB::TBinaryFeaturesPack> projBinaryFeatureValues(
        objectsDataProvider.GetBinaryFeaturesPacksSize(),
        NCB::TBinaryFeaturesPack(0));

    for (const TBinFeature& feature : proj.BinFeatures) {
        auto floatFeatureIdx = NCB::TFloatFeatureIdx((ui32)feature.FloatFeature);
        ProcessFeatureForCalcHashes<NCB::IQuantizedFloatValuesHolder>(
            objectsDataProvider.GetFloatFeatureToPackedBinaryIndex(floatFeatureIdx),
            featuresSubsetIndexing,
            /*processBinaryInPacks*/ processBinaryFeaturesInPacks,
            /*isBinaryFeatureEquals1*/ 1,
            binaryFeaturesBitMasks,
            projBinaryFeatureValues,
            [&]() { return *objectsDataProvider.GetFloatFeature(*floatFeatureIdx); },
            [&](ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
            [feature, hashArr] (ui32 i, ui8 featureValue) {
                const bool isTrueFeature = IsTrueHistogram(featureValue, (ui8)feature.SplitIdx);
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
            maybeBinaryIndex,
            featuresSubsetIndexing,
            /*processBinaryInPacks*/ processBinaryFeaturesInPacks,
            /*isBinaryFeatureEquals1*/ feature.Value == 1,
            binaryFeaturesBitMasks,
            projBinaryFeatureValues,
            [&]() { return *objectsDataProvider.GetCatFeature(*catFeatureIdx); },
            [&](ui32 packIdx) { return objectsDataProvider.GetBinaryFeaturesPack(packIdx); },
            [feature, hashArr, maxBin] (ui32 i, ui32 featureValue) {
                const bool isTrueFeature = IsTrueOneHotFeature(Min(featureValue, maxBin), (ui32)feature.Value);
                hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
            }
        );
    }

    if (processBinaryFeaturesInPacks) {
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
                        (ui64)((~(binaryFeaturesPack ^ packProjBinaryFeatureValues)) & bitMask) + (ui64)bitMask);
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
