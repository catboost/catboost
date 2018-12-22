#include "model_dataset_compatibility.h"

#include <util/generic/cast.h>
#include <util/generic/hash_set.h>


namespace NCB {

    static inline TString GetFeatureName(const TString& featureId, int featureIndex) {
        return featureId == "" ? ToString(featureIndex) : featureId;
    }

    bool CheckColumnRemappingPossible(
        const TFullModel& model,
        const TFeaturesLayout& datasetFeaturesLayout,
        const THashSet<ui32>& datasetCatFeatureFlatIndexes,
        THashMap<ui32, ui32>* columnIndexesReorderMap)
    {
        columnIndexesReorderMap->clear();
        THashSet<TString> modelFeatureIdSet;
        for (const TCatFeature& feature : model.ObliviousTrees.CatFeatures) {
            if (!feature.UsedInModel) {
                continue;
            }
            modelFeatureIdSet.insert(feature.FeatureId);
        }
        for (const TFloatFeature& floatFeature : model.ObliviousTrees.FloatFeatures) {
            if (!floatFeature.UsedInModel()) {
                continue;
            }
            modelFeatureIdSet.insert(floatFeature.FeatureId);
        }
        size_t featureNameIntersection = 0;
        THashMap<TString, ui32> datasetFeatureNamesMap;

        const auto& datasetFeaturesMetaInfo = datasetFeaturesLayout.GetExternalFeaturesMetaInfo();
        for (ui32 i = 0; i < (ui32)datasetFeaturesMetaInfo.size(); ++i) {
            featureNameIntersection += modelFeatureIdSet.contains(datasetFeaturesMetaInfo[i].Name);
            datasetFeatureNamesMap[datasetFeaturesMetaInfo[i].Name] = i;
        }
        // if we have unique feature names for all features in model and in pool we can fill column index reordering map if needed
        if (modelFeatureIdSet.size() != model.GetUsedCatFeaturesCount() + model.GetUsedFloatFeaturesCount()
                || (datasetFeatureNamesMap.size() !=
                        (size_t)datasetFeaturesLayout.GetExternalFeatureCount())
                || featureNameIntersection != modelFeatureIdSet.size())
        {
            return false;
        }
        bool needRemapping = false;
        for (const TCatFeature& feature : model.ObliviousTrees.CatFeatures) {
            if (!feature.UsedInModel) {
                continue;
            }
            const auto datasetFlatFeatureIndex = datasetFeatureNamesMap.at(feature.FeatureId);
            CB_ENSURE(
                datasetCatFeatureFlatIndexes.contains(datasetFlatFeatureIndex),
                "Feature " << feature.FeatureId << " is categorical in model but marked as numerical in dataset");
            (*columnIndexesReorderMap)[feature.FlatFeatureIndex] = datasetFlatFeatureIndex;
            needRemapping |= (datasetFlatFeatureIndex != SafeIntegerCast<ui32>(feature.FlatFeatureIndex));
        }
        for (const TFloatFeature& feature : model.ObliviousTrees.FloatFeatures) {
            if (!feature.UsedInModel()) {
                continue;
            }
            const auto datasetFlatFeatureIndex = datasetFeatureNamesMap.at(feature.FeatureId);
            CB_ENSURE(
                !datasetCatFeatureFlatIndexes.contains(datasetFlatFeatureIndex),
                "Feature " << feature.FeatureId << " is numerical in model but marked as categorical in dataset");
            (*columnIndexesReorderMap)[feature.FlatFeatureIndex] = datasetFlatFeatureIndex;
            needRemapping |= (datasetFlatFeatureIndex != SafeIntegerCast<ui32>(feature.FlatFeatureIndex));
        }
        if (!needRemapping) {
            columnIndexesReorderMap->clear();
        }
        return true;
    }

    void CheckModelAndDatasetCompatibility(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData,
        THashMap<ui32, ui32>* columnIndexesReorderMap)
    {
        // TODO(akhropov): Quantized datasets support. MLTOOLS-2582.
        CB_ENSURE(
            dynamic_cast<const TRawObjectsDataProvider*>(&objectsData),
            "Non-raw pools are incompatible with models"
        );

        const auto& datasetFeaturesLayout = *objectsData.GetFeaturesLayout();

        const auto datasetCatFeatureInternalIdxToExternalIdx =
            datasetFeaturesLayout.GetCatFeatureInternalIdxToExternalIdx();

        THashSet<ui32> datasetCatFeatures(
            datasetCatFeatureInternalIdxToExternalIdx.begin(),
            datasetCatFeatureInternalIdxToExternalIdx.end());

        if (CheckColumnRemappingPossible(
            model,
            datasetFeaturesLayout,
            datasetCatFeatures,
            columnIndexesReorderMap))
        {
            return;
        }

        const auto& datasetFeaturesMetaInfo = datasetFeaturesLayout.GetExternalFeaturesMetaInfo();

        for (const TCatFeature& catFeature : model.ObliviousTrees.CatFeatures) {
            if (!catFeature.UsedInModel) {
                continue;
            }
            TString featureModelName = GetFeatureName(catFeature.FeatureId, catFeature.FlatFeatureIndex);
            CB_ENSURE(
                SafeIntegerCast<size_t>(catFeature.FlatFeatureIndex) < datasetFeaturesMetaInfo.size(),
                "Feature " << featureModelName << " is present in model but not in pool.");
            if (SafeIntegerCast<size_t>(catFeature.FlatFeatureIndex) < datasetFeaturesMetaInfo.size()
                    && catFeature.FeatureId != ""
                    && datasetFeaturesMetaInfo[catFeature.FlatFeatureIndex].Name != "")
            {
                CB_ENSURE(
                    catFeature.FeatureId == datasetFeaturesMetaInfo[catFeature.FlatFeatureIndex].Name,
                    "Feature " << datasetFeaturesMetaInfo[catFeature.FlatFeatureIndex].Name
                    << " from pool must be " << catFeature.FeatureId << ".";);
            }
            TString featurePoolName;
            if (SafeIntegerCast<size_t>(catFeature.FlatFeatureIndex) < datasetFeaturesMetaInfo.size()) {
                featurePoolName = GetFeatureName(
                    datasetFeaturesMetaInfo[catFeature.FlatFeatureIndex].Name,
                    catFeature.FlatFeatureIndex);
            } else {
                featurePoolName = GetFeatureName("", catFeature.FlatFeatureIndex);
            }
            CB_ENSURE(
                datasetCatFeatures.contains(catFeature.FlatFeatureIndex),
                "Feature " << featurePoolName << " from pool must be categorical.");
        }

        for (const TFloatFeature& floatFeature : model.ObliviousTrees.FloatFeatures) {
            if (!floatFeature.UsedInModel()) {
                continue;
            }
            TString featureModelName = GetFeatureName(floatFeature.FeatureId, floatFeature.FlatFeatureIndex);
            CB_ENSURE(
                SafeIntegerCast<size_t>(floatFeature.FlatFeatureIndex) < datasetFeaturesMetaInfo.size(),
                "Feature " << featureModelName << " is present in model but not in pool.");
            if (SafeIntegerCast<size_t>(floatFeature.FlatFeatureIndex) < datasetFeaturesMetaInfo.size()
                    && floatFeature.FeatureId != ""
                    && datasetFeaturesMetaInfo[floatFeature.FlatFeatureIndex].Name != "")
            {
                CB_ENSURE(
                    floatFeature.FeatureId == datasetFeaturesMetaInfo[floatFeature.FlatFeatureIndex].Name,
                    "Feature " << datasetFeaturesMetaInfo[floatFeature.FlatFeatureIndex].Name
                    << " from pool must be " << floatFeature.FeatureId << ".");
            }
            TString featurePoolName;
            if (SafeIntegerCast<size_t>(floatFeature.FlatFeatureIndex) < datasetFeaturesMetaInfo.size()) {
                featurePoolName = GetFeatureName(
                    datasetFeaturesMetaInfo[floatFeature.FlatFeatureIndex].Name,
                    floatFeature.FlatFeatureIndex);
            } else {
                featurePoolName = GetFeatureName("", floatFeature.FlatFeatureIndex);
            }
            CB_ENSURE(
                !datasetCatFeatures.contains(floatFeature.FlatFeatureIndex),
                "Feature " << featurePoolName << " from pool must not be categorical.");
        }
    }

}
