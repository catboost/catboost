#include "cat_feature_perfect_hash_helper.h"

namespace NCatboostCuda {


    TVector<ui32>
    TCatFeaturesPerfectHashHelper::UpdatePerfectHashAndBinarize(ui32 dataProviderId, const float* hashesFloat,
                                                                ui32 hashesSize)  {
        const auto* hashes = reinterpret_cast<const int*>(hashesFloat);

        const ui32 featureId = FeaturesManager.GetFeatureManagerIdForCatFeature(dataProviderId);
        auto& featuresHash = FeaturesManager.CatFeaturesPerfectHash;

        TMap<int, ui32> binarization;
        {
            TGuard<TAdaptiveLock> guard(UpdateLock);
            if (!featuresHash.HasHashInRam) {
                featuresHash.Load();
            }
            binarization.swap(featuresHash.FeaturesPerfectHash[featureId]);
        }

        TVector<ui32> bins(hashesSize, 0);
        for (ui32 i = 0; i < hashesSize; ++i) {
            auto hash = hashes[i];
            if (binarization.count(hash) == 0) {
                binarization[hash] = (unsigned int)binarization.size();
            }
            bins[i] = binarization[hash];
        }

        if (binarization.size() > 1) {
            TGuard<TAdaptiveLock> guard(UpdateLock);
            const ui32 uniqueValues = binarization.size();
            featuresHash.FeaturesPerfectHash[featureId].swap(binarization);
            featuresHash.CatFeatureUniqueValues[featureId] = uniqueValues;
        }
        return bins;
    }


}
