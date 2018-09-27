#pragma once

#include "feature_index.h"

#include <catboost/libs/helpers/exception.h>
#include <util/generic/map.h>
#include <util/generic/vector.h>
#include <util/stream/file.h>
#include <util/system/fs.h>
#include <util/system/spinlock.h>
#include <util/system/tempfile.h>
#include <util/system/types.h>
#include <util/ysaveload.h>

namespace NCB {

    class TCatFeaturesPerfectHash {
    public:
        TCatFeaturesPerfectHash(ui32 catFeatureCount, const TString& storageFile)
            : StorageTempFile(storageFile)
            , CatFeatureUniqueValues(catFeatureCount, 0)
            , FeaturesPerfectHash(catFeatureCount)
        {
            HasHashInRam = true;
        }

        ~TCatFeaturesPerfectHash() = default;

        const TMap<int, ui32>& GetFeatureIndex(const TCatFeatureIdx catFeatureIdx) const {
            CheckHasFeature(catFeatureIdx);
            if (!HasHashInRam) {
                Load();
            }
            return FeaturesPerfectHash[*catFeatureIdx];
        }

        ui32 GetUniqueValues(const TCatFeatureIdx catFeatureIdx) const {
            CheckHasFeature(catFeatureIdx);
            const ui32 uniqueValues = CatFeatureUniqueValues[*catFeatureIdx];
            return uniqueValues > 1 ? uniqueValues : 0;
        }

        bool HasFeature(const TCatFeatureIdx catFeatureIdx) const {
            return (size_t)*catFeatureIdx < CatFeatureUniqueValues.size();
        }

        void FreeRam() const {
            Save();
            TVector<TMap<int, ui32>> empty;
            FeaturesPerfectHash.swap(empty);
            HasHashInRam = false;
        }

        Y_SAVELOAD_DEFINE(CatFeatureUniqueValues, FeaturesPerfectHash, HasHashInRam);

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
        TVector<ui32> CatFeatureUniqueValues; // [catFeatureIdx]
        mutable TVector<TMap<int, ui32>> FeaturesPerfectHash; // [catFeatureIdx]
        mutable bool HasHashInRam = true;
    };
}
