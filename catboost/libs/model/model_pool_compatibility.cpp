#include "model_pool_compatibility.h"

#include <util/generic/hash_set.h>

static inline TString GetFeatureName(const TString& featureId, int featureIndex) {
    return featureId == "" ? ToString(featureIndex) : featureId;
}

void CheckModelAndPoolCompatibility(const TFullModel& model, const TPool& pool) {
    const int poolFeaturesCount = pool.Docs.GetEffectiveFactorCount();
    THashSet<int> poolCatFeatures(pool.CatFeatures.begin(), pool.CatFeatures.end());

    for (const TCatFeature& catFeature : model.ObliviousTrees.CatFeatures) {
        TString featureModelName = GetFeatureName(catFeature.FeatureId, catFeature.FlatFeatureIndex);
        CB_ENSURE(
            catFeature.FlatFeatureIndex < poolFeaturesCount,
            "Feature " << featureModelName << " is present in model but not in pool."
        );
        if (catFeature.FlatFeatureIndex < pool.FeatureId.ysize()
            && catFeature.FeatureId != ""
            && pool.FeatureId[catFeature.FlatFeatureIndex] != ""
        ) {
            CB_ENSURE(
                catFeature.FeatureId == pool.FeatureId[catFeature.FlatFeatureIndex],
                "Feature " << pool.FeatureId[catFeature.FlatFeatureIndex]
                << " from pool must be " << catFeature.FeatureId << ".";
            );
        }
        TString featurePoolName;
        if (catFeature.FlatFeatureIndex < pool.FeatureId.ysize()) {
            featurePoolName = GetFeatureName(
                pool.FeatureId[catFeature.FlatFeatureIndex],
                catFeature.FlatFeatureIndex
            );
        } else {
            featurePoolName = GetFeatureName("", catFeature.FlatFeatureIndex);
        }
        CB_ENSURE(
            poolCatFeatures.has(catFeature.FlatFeatureIndex),
            "Feature " << featurePoolName << " from pool must be categorical."
        );
    }

    for (const TFloatFeature& floatFeature : model.ObliviousTrees.FloatFeatures) {
        TString featureModelName = GetFeatureName(floatFeature.FeatureId, floatFeature.FlatFeatureIndex);
        CB_ENSURE(
            floatFeature.FlatFeatureIndex < poolFeaturesCount,
            "Feature " << featureModelName << " is present in model but not in pool."
        );
        if (floatFeature.FlatFeatureIndex < pool.FeatureId.ysize()
            && floatFeature.FeatureId != ""
            && pool.FeatureId[floatFeature.FlatFeatureIndex] != ""
        ) {
            CB_ENSURE(
                floatFeature.FeatureId == pool.FeatureId[floatFeature.FlatFeatureIndex],
                "Feature " << pool.FeatureId[floatFeature.FlatFeatureIndex]
                << " from pool must be " << floatFeature.FeatureId << ".";
            );
        }
        TString featurePoolName;
        if (floatFeature.FlatFeatureIndex < pool.FeatureId.ysize()) {
            featurePoolName = GetFeatureName(
                pool.FeatureId[floatFeature.FlatFeatureIndex],
                floatFeature.FlatFeatureIndex
            );
        } else {
            featurePoolName = GetFeatureName("", floatFeature.FlatFeatureIndex);
        }
        CB_ENSURE(
            !poolCatFeatures.has(floatFeature.FlatFeatureIndex),
            "Feature " << featurePoolName << " from pool must not be categorical."
        );
    }
}
