#include "model_pool_compatibility.h"

#include <util/generic/hash_set.h>

static inline TString GetFeatureName(const TString& featureId, int featureIndex) {
    return featureId == "" ? ToString(featureIndex) : featureId;
}

bool CheckColumnRemappingPossible(const TFullModel& model, const TPool& pool, const THashSet<int>& poolCatFeatureFlatIndexes, THashMap<int, int>* columnIndexesReorderMap) {
    columnIndexesReorderMap->clear();
    THashSet<TString> modelFeatureIdSet;
    for (const TCatFeature& feature : model.ObliviousTrees.CatFeatures) {
        modelFeatureIdSet.insert(feature.FeatureId);
    }
    for (const TFloatFeature& floatFeature : model.ObliviousTrees.FloatFeatures) {
        modelFeatureIdSet.insert(floatFeature.FeatureId);
    }
    size_t featureNameIntersection = 0;
    THashMap<TString, int> poolFeatureNamesMap;
    for (int i = 0; i < pool.FeatureId.ysize(); ++i) {
        featureNameIntersection += modelFeatureIdSet.has(pool.FeatureId[i]);
        poolFeatureNamesMap[pool.FeatureId[i]] = i;
    }
    // if we have unique feature names for all features in model and in pool we can fill column index reordering map if needed
    if (modelFeatureIdSet.size() != model.ObliviousTrees.CatFeatures.size() + model.ObliviousTrees.FloatFeatures.size() ||
        poolFeatureNamesMap.ysize() != pool.GetFactorCount() ||
        featureNameIntersection != modelFeatureIdSet.size()
        ) {
        return false;
    }
    bool needRemapping = false;
    for (const TCatFeature& feature : model.ObliviousTrees.CatFeatures) {
        const auto poolFlatFeatureIndex = poolFeatureNamesMap.at(feature.FeatureId);
        CB_ENSURE(poolCatFeatureFlatIndexes.has(poolFlatFeatureIndex), "Feature " << feature.FeatureId << " is categorical in model but marked as numerical in dataset");
        (*columnIndexesReorderMap)[feature.FlatFeatureIndex] = poolFlatFeatureIndex;
        needRemapping |= (poolFlatFeatureIndex != feature.FlatFeatureIndex);
    }
    for (const TFloatFeature& feature : model.ObliviousTrees.FloatFeatures) {
        const auto poolFlatFeatureIndex = poolFeatureNamesMap.at(feature.FeatureId);
        CB_ENSURE(!poolCatFeatureFlatIndexes.has(poolFlatFeatureIndex), "Feature " << feature.FeatureId << " is numerical in model but marked as categorical in dataset");
        (*columnIndexesReorderMap)[feature.FlatFeatureIndex] = poolFlatFeatureIndex;
        needRemapping |= (poolFlatFeatureIndex != feature.FlatFeatureIndex);
    }
    if (!needRemapping) {
        columnIndexesReorderMap->clear();
    }
    return true;
}

void CheckModelAndPoolCompatibility(const TFullModel& model, const TPool& pool, THashMap<int, int>* columnIndexesReorderMap) {
    THashSet<int> poolCatFeatures(pool.CatFeatures.begin(), pool.CatFeatures.end());
    if (CheckColumnRemappingPossible(model, pool, poolCatFeatures, columnIndexesReorderMap)) {
        return;
    }
    const int poolFeaturesCount = pool.Docs.GetEffectiveFactorCount();

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
