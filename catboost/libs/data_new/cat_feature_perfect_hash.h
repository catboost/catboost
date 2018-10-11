#pragma once

#include "feature_index.h"

#include <catboost/libs/helpers/exception.h>
#include <util/generic/map.h>
#include <util/generic/typetraits.h>
#include <util/generic/vector.h>
#include <util/stream/file.h>
#include <util/system/fs.h>
#include <util/system/spinlock.h>
#include <util/system/tempfile.h>
#include <util/system/types.h>
#include <util/ysaveload.h>


namespace NCB {
    struct TCatFeatureUniqueValuesCounts {
        ui32 OnLearnOnly = 0;
        ui32 OnAll = 0;

    public:
        bool operator==(const TCatFeatureUniqueValuesCounts rhs) const {
            return (OnLearnOnly == rhs.OnLearnOnly) && (OnAll == rhs.OnAll);
        }
    };
}

Y_DECLARE_PODTYPE(NCB::TCatFeatureUniqueValuesCounts);


namespace NCB {

    class TCatFeaturesPerfectHash {
    public:
        TCatFeaturesPerfectHash(ui32 catFeatureCount, const TString& storageFile)
            : StorageTempFile(storageFile)
            , CatFeatureUniqValuesCountsVector(catFeatureCount)
            , FeaturesPerfectHash(catFeatureCount)
        {
            HasHashInRam = true;
        }

        ~TCatFeaturesPerfectHash() = default;

        bool operator==(const TCatFeaturesPerfectHash& rhs) const;

        const TMap<ui32, ui32>& GetFeaturePerfectHash(const TCatFeatureIdx catFeatureIdx) const {
            CheckHasFeature(catFeatureIdx);
            if (!HasHashInRam) {
                Load();
            }
            return FeaturesPerfectHash[*catFeatureIdx];
        }

        // for testing or setting from external sources
        void UpdateFeaturePerfectHash(const TCatFeatureIdx catFeatureIdx, TMap<ui32, ui32>&& perfectHash);

        TCatFeatureUniqueValuesCounts GetUniqueValuesCounts(const TCatFeatureIdx catFeatureIdx) const {
            CheckHasFeature(catFeatureIdx);
            const auto uniqValuesCounts = CatFeatureUniqValuesCountsVector[*catFeatureIdx];
            return uniqValuesCounts.OnAll > 1 ? uniqValuesCounts : TCatFeatureUniqueValuesCounts();
        }

        bool HasFeature(const TCatFeatureIdx catFeatureIdx) const {
            return (size_t)*catFeatureIdx < CatFeatureUniqValuesCountsVector.size();
        }

        void FreeRam() const {
            Save();
            TVector<TMap<ui32, ui32>> empty;
            FeaturesPerfectHash.swap(empty);
            HasHashInRam = false;
        }

        Y_SAVELOAD_DEFINE(CatFeatureUniqValuesCountsVector, FeaturesPerfectHash, HasHashInRam);

    private:
        void Save() const {
            TOFStream out(StorageTempFile.Name());
            ::Save(&out, FeaturesPerfectHash);
        }

        void Load() const {
            if (NFs::Exists(StorageTempFile.Name()) && !HasHashInRam) {
                TIFStream inputStream(StorageTempFile.Name());
                FeaturesPerfectHash.clear();
                ::Load(&inputStream, FeaturesPerfectHash);
                HasHashInRam = true;
            }
        }

    private:
        friend class TCatFeaturesPerfectHashHelper;

    private:
        void CheckHasFeature(const TCatFeatureIdx catFeatureIdx) const {
            CB_ENSURE_INTERNAL(
                HasFeature(catFeatureIdx),
                "Error: unknown categorical feature #" << catFeatureIdx
            );
        }

    private:
        TTempFile StorageTempFile;
        TVector<TCatFeatureUniqueValuesCounts> CatFeatureUniqValuesCountsVector; // [catFeatureIdx]
        mutable TVector<TMap<ui32, ui32>> FeaturesPerfectHash; // [catFeatureIdx]
        mutable bool HasHashInRam = true;
    };
}
