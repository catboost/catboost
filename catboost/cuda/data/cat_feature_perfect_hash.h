#pragma once

#include <catboost/libs/helpers/exception.h>
#include <util/generic/map.h>
#include <util/stream/file.h>
#include <util/system/fs.h>
#include <util/system/spinlock.h>
#include <util/system/tempfile.h>
#include <util/ysaveload.h>

namespace NCatboostCuda {
    class TCatFeaturesPerfectHash {
    public:
        explicit TCatFeaturesPerfectHash(const TString& storageFile)
            : StorageTempFile(storageFile)
        {
            HasHashInRam = true;
        }

        ~TCatFeaturesPerfectHash() = default;

        const TMap<int, ui32>& GetFeatureIndex(ui32 featureId) const {
            if (!HasHashInRam) {
                Load();
            }
            CB_ENSURE(FeaturesPerfectHash.has(featureId), "Features #" << featureId << " hash was not found");
            return FeaturesPerfectHash.at(featureId);
        }

        void RegisterId(ui32 featureId) {
            CB_ENSURE(HasHashInRam, "Can't register new features if hash is stored in file");
            FeaturesPerfectHash[featureId] = TMap<int, ui32>();
            CatFeatureUniqueValues[featureId] = 0;
        }

        ui32 GetUniqueValues(const ui32 featureId) const {
            if (CatFeatureUniqueValues.has(featureId)) {
                const ui32 uniqueValues = CatFeatureUniqueValues.at(featureId);
                return uniqueValues > 1 ? uniqueValues : 0;
            } else {
                ythrow TCatboostException() << "Error: unknown feature with id " << featureId;
            }
        }

        bool HasFeature(const ui32 featureId) const {
            return CatFeatureUniqueValues.has(featureId);
        }

        void FreeRam() const {
            Save();
            TMap<ui32, TMap<int, ui32>> empty;
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
        TTempFile StorageTempFile;
        TMap<ui32, ui32> CatFeatureUniqueValues;
        mutable TMap<ui32, TMap<int, ui32>> FeaturesPerfectHash;
        mutable bool HasHashInRam = true;
    };
}
