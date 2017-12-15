#pragma once

#include <catboost/cuda/data/binarizations_manager.h>
namespace NCatboostCuda {
    class TExternalCatFeatureValuesHolder: public ICatFeatureValuesHolder {
    public:
        TExternalCatFeatureValuesHolder(ui32 featureId,
                                        ui64 size,
                                        const float* featureHashes,
                                        const ui32 uniqueValues,
                                        const TBinarizedFeaturesManager& featuresManager,
                                        TString featureName = "")
            : ICatFeatureValuesHolder(featureId, size, std::move(featureName))
            , HashesPtr(reinterpret_cast<const int*>(featureHashes))
            , FeaturesManager(featuresManager)
            , UniqueValues(uniqueValues)
        {
            FeatureManagerFeatureId = FeaturesManager.GetFeatureManagerId(*this);
        }

        ui32 GetUniqueValues() const override {
            return UniqueValues;
        }

        ui32 GetValue(ui32 line) const override {
            CB_ENSURE(line < GetSize(), "Error: out of bounds");
            return GetBinFromHash(HashesPtr[line]);
        }

        TVector<ui32> ExtractValues() const override {
            TVector<ui32> values(GetSize());
            const auto& perfectHash = FeaturesManager.GetCategoricalFeaturesPerfectHash(FeatureManagerFeatureId);
            for (ui32 i = 0; i < values.size(); ++i) {
                CB_ENSURE(perfectHash.has(HashesPtr[i]),
                          "Error: hash for feature #" << FeatureManagerFeatureId << " was not found " << HashesPtr[i]);
                values[i] = perfectHash.at(HashesPtr[i]);
            }
            return values;
        }

    private:
        ui32 GetBinFromHash(int hash) const {
            const auto& perfectHash = FeaturesManager.GetCategoricalFeaturesPerfectHash(FeatureManagerFeatureId);
            return perfectHash.at(hash);
        }

    private:
        const int* HashesPtr;
        const TBinarizedFeaturesManager& FeaturesManager;
        ui32 UniqueValues;
        ui32 FeatureManagerFeatureId = -1;
    };
}
