#pragma once

#include <catboost/libs/data/exclusive_feature_bundling.h>
#include <catboost/libs/data/objects.h>
#include <catboost/libs/helpers/clear_array.h>

#include <library/containers/dense_hash/dense_hash.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/system/yassert.h>

#include <array>
#include <functional>


struct TProjection;


struct TCalcHashInBundleContext {
    ui32 InBundleIdx = 0;
    std::function<void(ui32, ui32)> CalcHashCallback;
};

struct TCalcHashInGroupContext {
    ui32 InGroupIdx;
    std::function<void(ui32, ui32)> CalcHashCallback;
};


template <class T, NCB::EFeatureValuesType FeatureValuesType, class F> // F args are (index, value)
inline void ProcessColumnForCalcHashes(
    const NCB::TTypedFeatureValuesHolder<T, FeatureValuesType>& column,
    const NCB::TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    F&& f,
    NPar::TLocalExecutor* localExecutor) {

    using TDenseHolder = NCB::TCompressedValuesHolderImpl<T, FeatureValuesType>;

    if (const auto* denseColumnData = dynamic_cast<const TDenseHolder*>(&column)) {
        const TCompressedArray& compressedArray = *denseColumnData->GetCompressedData().GetSrc();

        NCB::DispatchBitsPerKeyToDataType(
            compressedArray,
            "ProcessColumnForCalcHashes",
            [&] (const auto* histogram) {
                featuresSubsetIndexing.ParallelForEach(
                    [histogram, f] (ui32 i, ui32 srcIdx) {
                        f(i, histogram[srcIdx]);
                    },
                    localExecutor,
                    /*approximateBlockSize*/1000
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
    F&& f,
    NPar::TLocalExecutor* localExecutor) {

    ProcessColumnForCalcHashes(
        column,
        featuresSubsetIndexing,
        [f, getBinFromHistogramValue] (ui32 i, auto value) {
            f(i, getBinFromHistogramValue(value));
        },
        localExecutor);
}

template <class TValue, NCB::EFeatureValuesType FeatureValuesType, class TColumn>
inline void GetRawColumn(const TColumn& column, const void** rawPtr, ui32* bitsPerKey) {
    using TDenseHolder = NCB::TCompressedValuesHolderImpl<TValue, FeatureValuesType>;
    const auto* denseColumnData = dynamic_cast<const TDenseHolder*>(&column);
    CB_ENSURE(denseColumnData, "Wrong column type");
    const TCompressedArray& compressedArray = *denseColumnData->GetCompressedData().GetSrc();
    *rawPtr = (const void*)compressedArray.GetRawPtr();
    *bitsPerKey = compressedArray.GetBitsPerKey();
    CB_ENSURE_INTERNAL(
        *bitsPerKey <= sizeof(TValue) * 8, "BitsPerKey " << *bitsPerKey << " exceeds maximum width " << sizeof(TValue) * 8);
}

template <class T, NCB::EFeatureValuesType FeatureValuesType, class F>
inline void ProcessFeatureForCalcHashes(
    TMaybe<NCB::TExclusiveBundleIndex> maybeExclusiveBundleIndex,
    TMaybe<NCB::TPackedBinaryIndex> maybeBinaryIndex,
    TMaybe<NCB::TFeaturesGroupIndex> maybeFeaturesGroupIndex,
    const NCB::TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    bool processAggregatedFeatures,
    bool isBinaryFeatureEquals1, // used only if processBinary
    TArrayRef<TVector<TCalcHashInBundleContext>> featuresInBundles,
    TArrayRef<NCB::TBinaryFeaturesPack> binaryFeaturesBitMasks,
    TArrayRef<NCB::TBinaryFeaturesPack> projBinaryFeatureValues,
    TArrayRef<TVector<TCalcHashInGroupContext>> featuresInGroups,
    std::function<const NCB::TTypedFeatureValuesHolder<T, FeatureValuesType>*()>&& getFeatureColumn,
    std::function<const NCB::TExclusiveFeaturesBundle(ui32)>&& getExclusiveFeatureBundleMetaData,
    std::function<const NCB::TExclusiveFeatureBundleHolder*(ui32)>&& getExclusiveFeatureBundle,
    std::function<const NCB::TBinaryPacksHolder*(ui32)>&& getBinaryFeaturesPack,
    std::function<const NCB::TFeaturesGroupHolder*(ui32)>&& getFeaturesGroup,
    F&& f,
    NPar::TLocalExecutor* localExecutor) {

    if (maybeExclusiveBundleIndex) {
        if (processAggregatedFeatures) {
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
                std::move(f),
                localExecutor
            );
        }
    } else if (maybeBinaryIndex) {
        NCB::TBinaryFeaturesPack bitMask = NCB::TBinaryFeaturesPack(1) << maybeBinaryIndex->BitIdx;
        if (processAggregatedFeatures) {
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
                std::move(f),
                localExecutor
            );
        }
    } else if (maybeFeaturesGroupIndex) {
        if (processAggregatedFeatures) {
            TCalcHashInGroupContext calcHashInGroupContext;
            calcHashInGroupContext.InGroupIdx = maybeFeaturesGroupIndex->InGroupIdx;
            calcHashInGroupContext.CalcHashCallback = f;

            featuresInGroups[maybeFeaturesGroupIndex->GroupIdx].push_back(
                std::move(calcHashInGroupContext)
            );
        } else {
            const ui32 groupIdx = maybeFeaturesGroupIndex->GroupIdx;
            const ui32 partIdx = maybeFeaturesGroupIndex->InGroupIdx;

            ProcessColumnForCalcHashes(
                *getFeaturesGroup(groupIdx),
                featuresSubsetIndexing,
                [partIdx] (auto groupData) {
                    return NCB::GetPartValueFromGroup(groupData, partIdx);
                },
                std::move(f),
                localExecutor
            );
        }
    } else {
        ProcessColumnForCalcHashes(
            *getFeatureColumn(),
            featuresSubsetIndexing,
            [] (auto value) { return value; },
            std::move(f),
            localExecutor
        );
    }
}


template <class F>
inline void ExtractIndicesAndMasks(
    TMaybe<NCB::TExclusiveBundleIndex> maybeExclusiveBundleIndex,
    TMaybe<NCB::TPackedBinaryIndex> maybeBinaryIndex,
    TMaybe<NCB::TFeaturesGroupIndex> maybeFeaturesGroupIndex,
    bool isBinaryFeatureEquals1, // used only if processBinary
    TArrayRef<TVector<TCalcHashInBundleContext>> featuresInBundles,
    TArrayRef<NCB::TBinaryFeaturesPack> binaryFeaturesBitMasks,
    TArrayRef<NCB::TBinaryFeaturesPack> projBinaryFeatureValues,
    TArrayRef<TVector<TCalcHashInGroupContext>> featuresInGroups,
    F&& f
) {
    if (maybeExclusiveBundleIndex) {
        TCalcHashInBundleContext calcHashInBundleContext;
        calcHashInBundleContext.InBundleIdx = maybeExclusiveBundleIndex->InBundleIdx;
        calcHashInBundleContext.CalcHashCallback = f;
        featuresInBundles[maybeExclusiveBundleIndex->BundleIdx].push_back(
            std::move(calcHashInBundleContext)
        );
    } else if (maybeBinaryIndex) {
        NCB::TBinaryFeaturesPack bitMask = NCB::TBinaryFeaturesPack(1) << maybeBinaryIndex->BitIdx;
        binaryFeaturesBitMasks[maybeBinaryIndex->PackIdx] |= bitMask;
        if (isBinaryFeatureEquals1) {
            projBinaryFeatureValues[maybeBinaryIndex->PackIdx] |= bitMask;
        }
    } else if (maybeFeaturesGroupIndex) {
        TCalcHashInGroupContext calcHashInGroupContext;
        calcHashInGroupContext.InGroupIdx = maybeFeaturesGroupIndex->InGroupIdx;
        calcHashInGroupContext.CalcHashCallback = f;
        featuresInGroups[maybeFeaturesGroupIndex->GroupIdx].push_back(std::move(calcHashInGroupContext));
    }
}


/**
    Column formats and corresponding parameters for hash calculation
*/
struct TCalcHashParams {
    TMaybe<TConstArrayRef<ui32>> CatValuesDecoder;
    TMaybe<int> SplitIdx;
    TMaybe<std::array<ui32, 2>> MaxBinAndValue;

    TMaybe<NCB::TBoundsInBundle> Bounds;
    TMaybe<ui8> BitIdx;
    TMaybe<ui32> InGroupIdx;

    const void* RawColumnPtr = nullptr;
    ui32 BitsPerKey;

    void GatherValues(ui32 srcIdx, ui32 unrollCount, TArrayRef<ui64> values) const {
        if (BitsPerKey == 8) {
            for (ui32 unrollIdx : xrange(unrollCount)) {
                values[unrollIdx] = ((const ui8*)RawColumnPtr)[srcIdx + unrollIdx];
            }
        } else if (BitsPerKey == 16) {
            for (ui32 unrollIdx : xrange(unrollCount)) {
                values[unrollIdx] = ((const ui16*)RawColumnPtr)[srcIdx + unrollIdx];
            }
        } else {
            Y_ASSERT(BitsPerKey == 32);
            for (ui32 unrollIdx : xrange(unrollCount)) {
                values[unrollIdx] = ((const ui32*)RawColumnPtr)[srcIdx + unrollIdx];
            }
        }
        if (Bounds) {
            for (auto& value : values) {
                value = NCB::GetBinFromBundle<ui16>(value, Bounds.GetRef());
            }
        } else if (BitIdx) {
            for (auto& value : values) {
                value = (value >> BitIdx.GetRef()) & 1;
            }
        } else if (InGroupIdx) {
            for (auto& value : values) {
                value = NCB::GetPartValueFromGroup(value, *InGroupIdx);
            }
        }
    }

    void GatherValues(ui32 srcIdx, ui32 unrollCount, const ui32* srcIndices, TArrayRef<ui64> values) const {
        if (BitsPerKey == 8) {
            for (ui32 unrollIdx : xrange(unrollCount)) {
                values[unrollIdx] = ((const ui8*)RawColumnPtr)[srcIndices[srcIdx + unrollIdx]];
            }
        } else if (BitsPerKey == 16) {
            for (ui32 unrollIdx : xrange(unrollCount)) {
                values[unrollIdx] = ((const ui16*)RawColumnPtr)[srcIndices[srcIdx + unrollIdx]];
            }
        } else {
            Y_ASSERT(BitsPerKey == 32);
            for (ui32 unrollIdx : xrange(unrollCount)) {
                values[unrollIdx] = ((const ui32*)RawColumnPtr)[srcIndices[srcIdx + unrollIdx]];
            }
        }
        if (Bounds) {
            for (auto& value : values) {
                value = NCB::GetBinFromBundle<ui16>(value, Bounds.GetRef());
            }
        } else if (BitIdx) {
            for (auto& value : values) {
                value = (value >> BitIdx.GetRef()) & 1;
            }
        } else if (InGroupIdx) {
            for (auto& value : values) {
                value = NCB::GetPartValueFromGroup(value, *InGroupIdx);
            }
        }
    }
};

template <class T, NCB::EFeatureValuesType FeatureValuesType>
inline TCalcHashParams ExtractColumnLocation(
    TMaybe<NCB::TExclusiveBundleIndex> maybeExclusiveBundleIndex,
    TMaybe<NCB::TPackedBinaryIndex> maybeBinaryIndex,
    TMaybe<NCB::TFeaturesGroupIndex> maybeFeaturesGroupIndex,
    std::function<const NCB::TTypedFeatureValuesHolder<T, FeatureValuesType>*()>&& getFeatureColumn,
    std::function<const NCB::TExclusiveFeaturesBundle(ui32)>&& getExclusiveFeatureBundleMetaData,
    std::function<const NCB::TExclusiveFeatureBundleHolder*(ui32)>&& getExclusiveFeatureBundle,
    std::function<const NCB::TBinaryPacksHolder*(ui32)>&& getBinaryFeaturesPack,
    std::function<const NCB::TFeaturesGroupHolder*(ui32)>&& getFeaturesGroup
) {
    TCalcHashParams calcHashParams;
    if (maybeExclusiveBundleIndex) {
        const ui32 bundleIdx = maybeExclusiveBundleIndex->BundleIdx;
        GetRawColumn<ui16, NCB::EFeatureValuesType::ExclusiveFeatureBundle>(
            *getExclusiveFeatureBundle(bundleIdx),
            &calcHashParams.RawColumnPtr,
            &calcHashParams.BitsPerKey);
        const auto& metaData = getExclusiveFeatureBundleMetaData(bundleIdx);
        const auto boundsInBundle = metaData.Parts[maybeExclusiveBundleIndex->InBundleIdx].Bounds;
        calcHashParams.Bounds = boundsInBundle;
    } else if (maybeBinaryIndex) {
        GetRawColumn<ui8, NCB::EFeatureValuesType::BinaryPack>(
            *getBinaryFeaturesPack(maybeBinaryIndex->PackIdx),
            &calcHashParams.RawColumnPtr,
            &calcHashParams.BitsPerKey);
        calcHashParams.BitIdx = maybeBinaryIndex->BitIdx;
    } else if (maybeFeaturesGroupIndex) {
        const ui32 groupIdx = maybeFeaturesGroupIndex->GroupIdx;
        GetRawColumn<ui32, NCB::EFeatureValuesType::FeaturesGroup>(
            *getFeaturesGroup(groupIdx),
            &calcHashParams.RawColumnPtr,
            &calcHashParams.BitsPerKey);
        calcHashParams.InGroupIdx = maybeFeaturesGroupIndex->InGroupIdx;
    } else {
        GetRawColumn<T, FeatureValuesType>(
            *getFeatureColumn(),
            &calcHashParams.RawColumnPtr,
            &calcHashParams.BitsPerKey);
    }
    return calcHashParams;
}


/// Calculate document hashes into range [begin,end) for CTR bucket identification.
/// @param proj - Projection delivering the feature ids to hash
/// @param objectsDataProvider - Values of features to hash
/// @param featuresSubsetIndexing - Use these indices when accessing raw arrays data
/// @param perfectHashedToHashedCatValuesMap - if not nullptr use it to Hash original hashed cat values
//                                             if nullptr - used perfectHashed values
/// @param processAggregatedFeatures - process bundled, grouped and binary features in packs.
///                                       Faster, but not compatible with current model format.
///                                       So, enabled only during training, disabled for FinalCtr.
/// @param begin, @param end - Result range
void CalcHashes(
    const TProjection& proj,
    const NCB::TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const NCB::TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    const NCB::TPerfectHashedToHashedCatValuesMap* perfectHashedToHashedCatValuesMap,
    bool processAggregatedFeatures,
    ui64* begin,
    ui64* end,
    NPar::TLocalExecutor* localExecutor);


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
