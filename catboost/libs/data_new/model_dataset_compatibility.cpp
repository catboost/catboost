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
        for (const TCatFeature& feature : model.ObliviousTrees->CatFeatures) {
            if (!feature.UsedInModel) {
                continue;
            }
            modelFeatureIdSet.insert(feature.FeatureId);
        }
        for (const TFloatFeature& floatFeature : model.ObliviousTrees->FloatFeatures) {
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
        for (const TCatFeature& feature : model.ObliviousTrees->CatFeatures) {
            if (!feature.UsedInModel) {
                continue;
            }
            const auto datasetFlatFeatureIndex = datasetFeatureNamesMap.at(feature.FeatureId);
            CB_ENSURE(
                datasetCatFeatureFlatIndexes.contains(datasetFlatFeatureIndex),
                "Feature " << feature.FeatureId << " is categorical in model but marked as numerical in dataset");
            (*columnIndexesReorderMap)[feature.Position.FlatIndex] = datasetFlatFeatureIndex;
            needRemapping |= (datasetFlatFeatureIndex != SafeIntegerCast<ui32>(feature.Position.FlatIndex));
        }
        for (const TFloatFeature& feature : model.ObliviousTrees->FloatFeatures) {
            if (!feature.UsedInModel()) {
                continue;
            }
            const auto datasetFlatFeatureIndex = datasetFeatureNamesMap.at(feature.FeatureId);
            CB_ENSURE(
                !datasetCatFeatureFlatIndexes.contains(datasetFlatFeatureIndex),
                "Feature " << feature.FeatureId << " is numerical in model but marked as categorical in dataset");
            (*columnIndexesReorderMap)[feature.Position.FlatIndex] = datasetFlatFeatureIndex;
            needRemapping |= (datasetFlatFeatureIndex != SafeIntegerCast<ui32>(feature.Position.FlatIndex));
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
        if (dynamic_cast<const TQuantizedForCPUObjectsDataProvider*>(&objectsData)) {
            CB_ENSURE(model.GetUsedCatFeaturesCount() == 0, "Quantized datasets with categorical features are not currently supported");
        }
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

        for (const TCatFeature& catFeature : model.ObliviousTrees->CatFeatures) {
            if (!catFeature.UsedInModel) {
                continue;
            }
            TString featureModelName = GetFeatureName(catFeature.FeatureId, catFeature.Position.FlatIndex);
            CB_ENSURE(
                SafeIntegerCast<size_t>(catFeature.Position.FlatIndex) < datasetFeaturesMetaInfo.size(),
                "Feature " << featureModelName << " is present in model but not in pool.");
            if (SafeIntegerCast<size_t>(catFeature.Position.FlatIndex) < datasetFeaturesMetaInfo.size()
                    && catFeature.FeatureId != ""
                    && datasetFeaturesMetaInfo[catFeature.Position.FlatIndex].Name != "")
            {
                CB_ENSURE(
                    catFeature.FeatureId == datasetFeaturesMetaInfo[catFeature.Position.FlatIndex].Name,
                    "Feature " << datasetFeaturesMetaInfo[catFeature.Position.FlatIndex].Name
                    << " from pool must be " << catFeature.FeatureId << ".";);
            }
            TString featurePoolName;
            if (SafeIntegerCast<size_t>(catFeature.Position.FlatIndex) < datasetFeaturesMetaInfo.size()) {
                featurePoolName = GetFeatureName(
                    datasetFeaturesMetaInfo[catFeature.Position.FlatIndex].Name,
                    catFeature.Position.FlatIndex);
            } else {
                featurePoolName = GetFeatureName("", catFeature.Position.FlatIndex);
            }
            CB_ENSURE(
                datasetCatFeatures.contains(catFeature.Position.FlatIndex),
                "Feature " << featurePoolName << " from pool must be categorical.");
        }

        for (const TFloatFeature& floatFeature : model.ObliviousTrees->FloatFeatures) {
            if (!floatFeature.UsedInModel()) {
                continue;
            }
            TString featureModelName = GetFeatureName(floatFeature.FeatureId, floatFeature.Position.FlatIndex);
            CB_ENSURE(
                SafeIntegerCast<size_t>(floatFeature.Position.FlatIndex) < datasetFeaturesMetaInfo.size(),
                "Feature " << featureModelName << " is present in model but not in pool.");
            if (SafeIntegerCast<size_t>(floatFeature.Position.FlatIndex) < datasetFeaturesMetaInfo.size()
                    && floatFeature.FeatureId != ""
                    && datasetFeaturesMetaInfo[floatFeature.Position.FlatIndex].Name != "")
            {
                CB_ENSURE(
                    floatFeature.FeatureId == datasetFeaturesMetaInfo[floatFeature.Position.FlatIndex].Name,
                    "Feature " << datasetFeaturesMetaInfo[floatFeature.Position.FlatIndex].Name
                    << " from pool must be " << floatFeature.FeatureId << ".");
            }
            TString featurePoolName;
            if (SafeIntegerCast<size_t>(floatFeature.Position.FlatIndex) < datasetFeaturesMetaInfo.size()) {
                featurePoolName = GetFeatureName(
                    datasetFeaturesMetaInfo[floatFeature.Position.FlatIndex].Name,
                    floatFeature.Position.FlatIndex);
            } else {
                featurePoolName = GetFeatureName("", floatFeature.Position.FlatIndex);
            }
            CB_ENSURE(
                !datasetCatFeatures.contains(floatFeature.Position.FlatIndex),
                "Feature " << featurePoolName << " from pool must not be categorical.");
        }
    }

    TVector<TVector<ui8>> GetFloatFeaturesBordersRemap(
        const TFullModel& model,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo)
    {
        TVector<TVector<ui8>> floatBinsRemap(model.ObliviousTrees->FloatFeatures.size());
        for (const auto& feature: model.ObliviousTrees->FloatFeatures) {
            if (feature.Borders.empty()) {
                continue;
            }
            CB_ENSURE(
                quantizedFeaturesInfo.HasBorders(NCB::TFloatFeatureIdx(feature.Position.FlatIndex)),
                "Feature " << feature.Position.FlatIndex <<  ": dataset does not have border information for it"
            );
            auto& quantizedBorders = quantizedFeaturesInfo.GetBorders(NCB::TFloatFeatureIdx(feature.Position.FlatIndex));
            ui32 poolBucketIdx = 0;
            auto addRemapBinIdx = [&] (ui8 bucketIdx) {
                floatBinsRemap[feature.Position.FlatIndex].push_back(bucketIdx);
                ++poolBucketIdx;
            };
            for (ui32 modelBucketIdx = 0; modelBucketIdx < feature.Borders.size(); ++modelBucketIdx) {
                while (poolBucketIdx < quantizedBorders.size() &&
                    quantizedBorders[poolBucketIdx] < feature.Borders[modelBucketIdx]) {
                    addRemapBinIdx(modelBucketIdx);
                }
                CB_ENSURE(poolBucketIdx < quantizedBorders.size(),
                    "Feature " << feature.Position.FlatIndex << ": inconsistent borders, last quantized vs model: "
                    << double(quantizedBorders.back()) << " vs " << feature.Borders[modelBucketIdx]
                );
                CB_ENSURE(quantizedBorders[poolBucketIdx] == feature.Borders[modelBucketIdx],
                    "Feature " << feature.Position.FlatIndex << ": inconsistent borders, quantized vs model: "
                    << double(quantizedBorders[poolBucketIdx]) << " vs " << feature.Borders[modelBucketIdx]
                );
                addRemapBinIdx(modelBucketIdx);
            }
            while (poolBucketIdx <= quantizedBorders.size()) {
                addRemapBinIdx(feature.Borders.size());
            }
        }
        return floatBinsRemap;
    }
}
