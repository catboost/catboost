#pragma once

#include <util/system/fs.h>
#include <util/system/spinlock.h>


class TCatFeaturesPerfectHash {
public:
    explicit TCatFeaturesPerfectHash(const TString& storageFile)
        : StorageFile(storageFile) {
        HasHashInRam = true;

    }

    const ymap<int, ui32>& GetFeatureIndex(ui32 featureId) const {
        if (!HasHashInRam) {
            Load();
        }
        CB_ENSURE(FeaturesPerfectHash.has(featureId), "Features #" << featureId << " hash was not found");
        return FeaturesPerfectHash.at(featureId);
    }

    void RegisterId(ui32 featureId) {
        CB_ENSURE(HasHashInRam, "Can't register new features if hash is stored in file");
        FeaturesPerfectHash[featureId] = ymap<int, ui32>();
        CatFeatureUniqueValues[featureId] = 0;
    }

    ui32 GetUniqueValues(const ui32 featureId) const {
        if (CatFeatureUniqueValues.has(featureId)) {
            return CatFeatureUniqueValues.at(featureId);
        } else {
           ythrow TCatboostException() << "Error: unknown feature with id " << featureId;
        }
    }

    bool HasFeature(const ui32 featureId) const {
        return CatFeatureUniqueValues.has(featureId);
    }

    void FreeRam() const {
        Save();
        ymap<ui32, ymap<int, ui32>> empty;
        FeaturesPerfectHash.swap(empty);
        HasHashInRam = false;
    }

    SAVELOAD(StorageFile, CatFeatureUniqueValues, FeaturesPerfectHash, HasHashInRam);

private:

    void Save() const {
        TOFStream out(StorageFile);
        ::Save(&out, FeaturesPerfectHash);
    }

    void Load() const {
        if (NFs::Exists(StorageFile) && !HasHashInRam) {
            TIFStream inputStream(StorageFile);
            FeaturesPerfectHash.clear();
            ::Load(&inputStream, FeaturesPerfectHash);
            HasHashInRam = true;
        }
    }
private:
    friend class TCatFeaturesPerfectHashHelper;
private:
    TString StorageFile;
    ymap<ui32, ui32> CatFeatureUniqueValues;
    mutable ymap<ui32, ymap<int, ui32>> FeaturesPerfectHash;
    mutable bool HasHashInRam = true;
};
