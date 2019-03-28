#pragma once

#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/dbg_output.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/index_range/index_range.h>
#include <catboost/libs/options/enums.h>

#include <library/binsaver/bin_saver.h>
#include <library/dbg_output/dump.h>

#include <util/generic/array_ref.h>
#include <util/generic/bitops.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/system/types.h>

#include <climits>


namespace NPar {
    class TLocalExecutor;
}


namespace NCB {

    static_assert(CHAR_BIT == 8, "CatBoost requires CHAR_BIT == 8");

    struct TRawObjectsData;
    class TQuantizedFeaturesInfo;

    using TFeaturesArraySubsetIndexing = TArraySubsetIndexing<ui32>;


    /* values are shifted by 1 (because 0 is not stored)
     * , so [Begin, End) values in bundle means [Begin + 1, End + 1) bin values
     *  if all bins in bundle are 0 the bundle value will be Parts.back().End
     */
    using TBoundsInBundle = TIndexRange<ui32>;


    struct TExclusiveBundlePart {
        EFeatureType FeatureType;
        ui32 FeatureIdx; // per type
        TBoundsInBundle Bounds;

    public:
        // needed for BinSaver
        explicit TExclusiveBundlePart(
            EFeatureType featureType = EFeatureType::Float,
            ui32 featureIdx = 0,
            TBoundsInBundle bounds = TBoundsInBundle(0))
            : FeatureType(featureType)
            , FeatureIdx(featureIdx)
            , Bounds(bounds)
        {}

        bool operator==(const TExclusiveBundlePart& rhs) const {
            return (FeatureType == rhs.FeatureType) &&
                (FeatureIdx == rhs.FeatureIdx) &&
                (Bounds == rhs.Bounds);
        }

        SAVELOAD(FeatureType, FeatureIdx, Bounds);
    };


    struct TExclusiveFeaturesBundle {
        ui32 SizeInBytes = 0; // TODO(akhropov): do we need bits?
        TVector<TExclusiveBundlePart> Parts;

    public:
        SAVELOAD(SizeInBytes, Parts);

        bool operator==(const TExclusiveFeaturesBundle& rhs) const {
            return (SizeInBytes == rhs.SizeInBytes) && (Parts == rhs.Parts);
        }

        ui32 GetUsedByPartsBinCount() const {
            return Parts.empty() ? ui32(0) : Parts.back().Bounds.End;
        }

        // used by parts + one bin for default value (0)
        ui32 GetBinCount() const {
            return GetUsedByPartsBinCount() + 1;
        }

        bool IsBinaryFeaturesOnly() const {
            return (size_t)GetUsedByPartsBinCount() == Parts.size();
        }

        void Add(TExclusiveBundlePart&& part) {
            CB_ENSURE_INTERNAL(
                part.Bounds.Begin == GetUsedByPartsBinCount(),
                "Non-consecutive bounds in added bundle part"
            );
            Parts.push_back(std::move(part));
            SizeInBytes = CeilDiv(
                GetValueBitCount(GetUsedByPartsBinCount()),
                unsigned(CHAR_BIT)
            );
            CB_ENSURE_INTERNAL(SizeInBytes <= 2, "SizeInBytes > 2 is not currently supported");
        }
    };


    struct TExclusiveBundleIndex {
        ui32 BundleIdx;
        ui32 InBundleIdx;

    public:
        // default initialization needed for BinSaver
        explicit TExclusiveBundleIndex(
            ui32 bundleIdx = 0,
            ui32 inBundleIdx = 0)
            : BundleIdx(bundleIdx)
            , InBundleIdx(inBundleIdx)
        {}

        bool operator==(const TExclusiveBundleIndex& rhs) const {
            return (BundleIdx == rhs.BundleIdx) && (InBundleIdx == rhs.InBundleIdx);
        }

        SAVELOAD(BundleIdx, InBundleIdx);
    };


    template <class TBin, class TBundle>
    inline TBin GetBinFromBundle(TBundle bundle, TBoundsInBundle bounds) {
        if ((bundle < (TBundle)bounds.Begin) || (bundle >= (TBundle)bounds.End)) {
            return TBin(0);
        }
        return (TBin)(bundle - (TBundle)bounds.Begin + TBundle(1));
    }


    struct TFeaturesBundleArraySubset {
        const TExclusiveFeaturesBundle* MetaData;

        TConstArrayRef<ui8> SrcData;
        const TArraySubsetIndexing<ui32>* SubsetIndexing;

    public:
        TFeaturesBundleArraySubset(
            const TExclusiveFeaturesBundle* metaData,
            TConstArrayRef<ui8> srcData,
            const TArraySubsetIndexing<ui32>* subsetIndexing)
            : MetaData(metaData)
            , SrcData(srcData)
            , SubsetIndexing(subsetIndexing)
        {}
    };


    struct TExclusiveFeaturesBundlingOptions {
        ui32 MaxBuckets = 1 << 10;
        float MaxConflictFraction = 0.0f;
    };


    TVector<TExclusiveFeaturesBundle> CreateExclusiveFeatureBundles(
        const TRawObjectsData& rawObjectsData,
        const TFeaturesArraySubsetIndexing& rawDataSubsetIndexing,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TExclusiveFeaturesBundlingOptions& options,
        NPar::TLocalExecutor* localExecutor
    );
}


template <>
struct TDumper<NCB::TExclusiveBundlePart> {
    template <class S>
    static inline void Dump(S& s, const NCB::TExclusiveBundlePart& exclusiveBundlePart) {
        s << "FeatureType=" << exclusiveBundlePart.FeatureType
          << ",FeatureIdx=" << exclusiveBundlePart.FeatureIdx
          << ",BoundsInBundle=[" << exclusiveBundlePart.Bounds.Begin << ","
          << exclusiveBundlePart.Bounds.End << ")";
    }
};

template <>
struct TDumper<NCB::TExclusiveFeaturesBundle> {
    template <class S>
    static inline void Dump(S& s, const NCB::TExclusiveFeaturesBundle& exclusiveFeaturesBundle) {
        s << "SizeInBytes=" << exclusiveFeaturesBundle.SizeInBytes
          << ",Parts=[" << NCB::DbgDumpWithIndices(exclusiveFeaturesBundle.Parts) << "\n";
    }
};
