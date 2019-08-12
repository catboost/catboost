#pragma once

#include "feature_index.h"

#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/exception.h>

#include <library/binsaver/bin_saver.h>

#include <util/generic/guid.h>
#include <util/generic/map.h>
#include <util/generic/string.h>
#include <util/generic/typetraits.h>
#include <util/generic/vector.h>
#include <util/stream/file.h>
#include <util/system/fs.h>
#include <util/system/mktemp.h>
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

    // for some reason TCatFeatureUniqueValuesCounts is not std::is_trivial
    inline ui32 UpdateCheckSumImpl(ui32 init, const TCatFeatureUniqueValuesCounts& data) {
        ui32 checkSum = UpdateCheckSum(init, data.OnLearnOnly);
        return UpdateCheckSum(checkSum, data.OnAll);
    }

    struct TValueWithCount {
        ui32 Value = 0;
        ui32 Count = 0;

    public:
        bool operator==(const TValueWithCount rhs) const {
            return (Value == rhs.Value) && (Count == rhs.Count);
        }
    };

    // for some reason TValueWithCount is not std::is_trivial
    inline ui32 UpdateCheckSumImpl(ui32 init, const TValueWithCount& data) {
        ui32 checkSum = UpdateCheckSum(init, data.Value);
        return UpdateCheckSum(checkSum, data.Count);
    }

    using TCatFeaturePerfectHash = TMap<ui32, TValueWithCount>;
}

Y_DECLARE_PODTYPE(NCB::TCatFeatureUniqueValuesCounts);
Y_DECLARE_PODTYPE(NCB::TValueWithCount);


namespace NCB {

    class TCatFeaturesPerfectHash {
    public:
        // for IBinSaver
        TCatFeaturesPerfectHash()
            : StorageTempFile(TString::Join("cat_feature_index.", CreateGuidAsString(), ".tmp"))
        {}

        TCatFeaturesPerfectHash(ui32 catFeatureCount, const TString& storageFile, bool allowWriteFiles)
            : StorageTempFile(storageFile)
            , CatFeatureUniqValuesCountsVector(catFeatureCount)
            , FeaturesPerfectHash(catFeatureCount)
            , AllowWriteFiles(allowWriteFiles)
        {
            HasHashInRam = true;
        }

        ~TCatFeaturesPerfectHash() = default;

        bool operator==(const TCatFeaturesPerfectHash& rhs) const;

        // *this contains a superset of mapping in rhs
        bool IsSupersetOf(const TCatFeaturesPerfectHash& rhs) const;

        const TCatFeaturePerfectHash& GetFeaturePerfectHash(const TCatFeatureIdx catFeatureIdx) const {
            CheckHasFeature(catFeatureIdx);
            if (!HasHashInRam) {
                Load();
            }
            return FeaturesPerfectHash[*catFeatureIdx];
        }

        // for testing or setting from external sources
        void UpdateFeaturePerfectHash(const TCatFeatureIdx catFeatureIdx, TCatFeaturePerfectHash&& perfectHash);

        TCatFeatureUniqueValuesCounts GetUniqueValuesCounts(const TCatFeatureIdx catFeatureIdx) const {
            CheckHasFeature(catFeatureIdx);
            const auto uniqValuesCounts = CatFeatureUniqValuesCountsVector[*catFeatureIdx];
            return uniqValuesCounts.OnAll > 1 ? uniqValuesCounts : TCatFeatureUniqueValuesCounts();
        }

        bool HasFeature(const TCatFeatureIdx catFeatureIdx) const {
            return (size_t)*catFeatureIdx < CatFeatureUniqValuesCountsVector.size();
        }

        void SetAllowWriteFiles(bool allowWriteFiles) {
            AllowWriteFiles = allowWriteFiles;
        }

        void FreeRamIfPossible() const {
            if (AllowWriteFiles) {
                Save();
                TVector<TCatFeaturePerfectHash> empty;
                FeaturesPerfectHash.swap(empty);
                HasHashInRam = false;
            }
        }

        void Load() const {
            if (NFs::Exists(StorageTempFile.Name()) && !HasHashInRam) {
                TIFStream inputStream(StorageTempFile.Name());
                FeaturesPerfectHash.clear();
                ::Load(&inputStream, FeaturesPerfectHash);
                HasHashInRam = true;
            }
        }

        Y_SAVELOAD_DEFINE(CatFeatureUniqValuesCountsVector, FeaturesPerfectHash, HasHashInRam);

        int operator&(IBinSaver& binSaver);

        ui32 CalcCheckSum() const;

    private:
        void Save() const {
            TOFStream out(StorageTempFile.Name());
            ::Save(&out, FeaturesPerfectHash);
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
        mutable TVector<TCatFeaturePerfectHash> FeaturesPerfectHash; // [catFeatureIdx]
        mutable bool HasHashInRam = true;
        bool AllowWriteFiles = true;
    };
}
