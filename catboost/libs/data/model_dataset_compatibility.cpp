#include "model_dataset_compatibility.h"

#include <util/generic/cast.h>
#include <util/generic/hash_set.h>

template <class TFeature>
static void AddUsedFeatureIdsToSet(TConstArrayRef<TFeature> features, THashSet<TString>* featureIdSet) {
    for (const TFeature& feature : features) {
        if (!feature.UsedInModel()) {
            continue;
        }
        featureIdSet->insert(feature.FeatureId);
    }
}

template <class TFeature>
static constexpr const char* FeatureTypeName() {
    if constexpr (std::is_same<TFeature, TFloatFeature>::value) {
        return "Float";
    } else if constexpr (std::is_same<TFeature, TCatFeature>::value) {
        return "Categorical";
    } else if constexpr (std::is_same<TFeature, TTextFeature>::value) {
        return "Text";
    } else {
        CB_ENSURE(false, "FeatureTypeName: Unknown feature type");
        return "Unknown";
    }
}

template <class TFeature>
static void CheckFeatureTypes(
    TConstArrayRef<TFeature> features,
    const THashMap<TString, ui32>& datasetFeatureNamesMap,
    const THashSet<ui32>& datasetFeatureFlatIndexes,
    THashMap<ui32, ui32>* columnIndexesReorderMap
) {
    for (const TFeature& feature : features) {
        if (!feature.UsedInModel()) {
            continue;
        }
        const auto datasetFlatFeatureIndex = datasetFeatureNamesMap.at(feature.FeatureId);
        const TString featureTypeName = FeatureTypeName<TFeature>();
        CB_ENSURE(
            datasetFeatureFlatIndexes.contains(datasetFlatFeatureIndex),
            "Feature " << feature.FeatureId <<
                " is " << featureTypeName << " in model " <<
                "but marked different in the dataset");
        (*columnIndexesReorderMap)[feature.Position.FlatIndex] = datasetFlatFeatureIndex;
    }
}

static THashSet<ui32> GetFloatFeatureIndexes(
    ui32 featureCount,
    const THashSet<ui32>& catFeatureIndexes,
    const THashSet<ui32>& textFeatureIndexes
) {
    THashSet<ui32> allIndexes = xrange(featureCount);
    THashSet<ui32> floatFeatureIndexes;
    SetDifference(
        allIndexes.begin(),
        allIndexes.end(),
        catFeatureIndexes.begin(),
        catFeatureIndexes.end(),
        std::inserter(floatFeatureIndexes, floatFeatureIndexes.begin())
    );
    allIndexes = floatFeatureIndexes;
    SetDifference(
        allIndexes.begin(),
        allIndexes.end(),
        textFeatureIndexes.begin(),
        textFeatureIndexes.end(),
        std::inserter(floatFeatureIndexes, floatFeatureIndexes.begin())
    );
    return floatFeatureIndexes;
}

namespace NCB {

    static inline TString GetFeatureName(const TString& featureId, int featureIndex) {
        return featureId == "" ? ToString(featureIndex) : featureId;
    }

    bool CheckColumnRemappingPossible(
        const TFullModel& model,
        const TFeaturesLayout& datasetFeaturesLayout,
        const THashSet<ui32>& datasetCatFeatureFlatIndexes,
        const THashSet<ui32>& datasetTextFeatureFlatIndexes,
        THashMap<ui32, ui32>* columnIndexesReorderMap)
    {
        columnIndexesReorderMap->clear();
        THashSet<TString> modelFeatureIdSet;
        AddUsedFeatureIdsToSet(
            model.ModelTrees->GetFloatFeatures(),
            &modelFeatureIdSet
        );
        AddUsedFeatureIdsToSet(
            model.ModelTrees->GetCatFeatures(),
            &modelFeatureIdSet
        );
        AddUsedFeatureIdsToSet(
            model.ModelTrees->GetTextFeatures(),
            &modelFeatureIdSet
        );
        size_t featureNameIntersection = 0;
        THashMap<TString, ui32> datasetFeatureNamesMap;

        const auto& datasetFeaturesMetaInfo = datasetFeaturesLayout.GetExternalFeaturesMetaInfo();
        for (ui32 i = 0; i < (ui32)datasetFeaturesMetaInfo.size(); ++i) {
            featureNameIntersection += modelFeatureIdSet.contains(datasetFeaturesMetaInfo[i].Name);
            datasetFeatureNamesMap[datasetFeaturesMetaInfo[i].Name] = i;
        }
        // if we have unique feature names for all features in model and in pool we can fill column index reordering map if needed
        if (modelFeatureIdSet.size() !=
                model.GetUsedCatFeaturesCount() + model.GetUsedFloatFeaturesCount() + model.GetUsedTextFeaturesCount()
                || (datasetFeatureNamesMap.size() !=
                        (size_t)datasetFeaturesLayout.GetExternalFeatureCount())
                || featureNameIntersection != modelFeatureIdSet.size())
        {
            return false;
        }

        {
            const auto& datasetFloatFeatureIndexes = GetFloatFeatureIndexes(
                datasetFeaturesLayout.GetExternalFeatureCount(),
                datasetCatFeatureFlatIndexes,
                datasetTextFeatureFlatIndexes
            );
            CheckFeatureTypes(
                model.ModelTrees->GetFloatFeatures(),
                datasetFeatureNamesMap,
                datasetFloatFeatureIndexes,
                columnIndexesReorderMap
            );
        }
        CheckFeatureTypes(
            model.ModelTrees->GetCatFeatures(),
            datasetFeatureNamesMap,
            datasetCatFeatureFlatIndexes,
            columnIndexesReorderMap
        );
        CheckFeatureTypes(
            model.ModelTrees->GetTextFeatures(),
            datasetFeatureNamesMap,
            datasetTextFeatureFlatIndexes,
            columnIndexesReorderMap
        );
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

        const auto datasetTextFeatureInternalIdxToExternalIdx =
            datasetFeaturesLayout.GetTextFeatureInternalIdxToExternalIdx();

        THashSet<ui32> datasetTextFeatures(
            datasetTextFeatureInternalIdxToExternalIdx.begin(),
            datasetTextFeatureInternalIdxToExternalIdx.end());

        if (CheckColumnRemappingPossible(
            model,
            datasetFeaturesLayout,
            datasetCatFeatures,
            datasetTextFeatures,
            columnIndexesReorderMap))
        {
            return;
        }

        columnIndexesReorderMap->clear();

        const auto& datasetFeaturesMetaInfo = datasetFeaturesLayout.GetExternalFeaturesMetaInfo();

        for (const TCatFeature& catFeature : model.ModelTrees->GetCatFeatures()) {
            if (!catFeature.UsedInModel()) {
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

            columnIndexesReorderMap->insert(
                {catFeature.Position.FlatIndex, catFeature.Position.FlatIndex});
        }

        for (const TFloatFeature& floatFeature : model.ModelTrees->GetFloatFeatures()) {
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
            CB_ENSURE(
                !datasetTextFeatures.contains(floatFeature.Position.FlatIndex),
                "Feature " << featurePoolName << " from pool must not be text.");

            columnIndexesReorderMap->insert(
                {floatFeature.Position.FlatIndex, floatFeature.Position.FlatIndex});
        }

        // TODO(d-kruchinin, akhropov): refactor this - remove duplicates, create generic solution
        for (const TTextFeature& textFeature : model.ModelTrees->GetTextFeatures()) {
            if (!textFeature.UsedInModel()) {
                continue;
            }
            TString featureModelName = GetFeatureName(textFeature.FeatureId, textFeature.Position.FlatIndex);
            CB_ENSURE(
                SafeIntegerCast<size_t>(textFeature.Position.FlatIndex) < datasetFeaturesMetaInfo.size(),
                "Feature " << featureModelName << " is present in model but not in pool.");
            if (SafeIntegerCast<size_t>(textFeature.Position.FlatIndex) < datasetFeaturesMetaInfo.size()
                && textFeature.FeatureId != ""
                && datasetFeaturesMetaInfo[textFeature.Position.FlatIndex].Name != "")
            {
                CB_ENSURE(
                    textFeature.FeatureId == datasetFeaturesMetaInfo[textFeature.Position.FlatIndex].Name,
                    "Feature " << datasetFeaturesMetaInfo[textFeature.Position.FlatIndex].Name
                    << " from pool must be " << textFeature.FeatureId << ".");
            }
            TString featurePoolName;
            if (SafeIntegerCast<size_t>(textFeature.Position.FlatIndex) < datasetFeaturesMetaInfo.size()) {
                featurePoolName = GetFeatureName(
                    datasetFeaturesMetaInfo[textFeature.Position.FlatIndex].Name,
                    textFeature.Position.FlatIndex);
            } else {
                featurePoolName = GetFeatureName("", textFeature.Position.FlatIndex);
            }
            CB_ENSURE(
                datasetTextFeatures.contains(textFeature.Position.FlatIndex),
                "Feature " << featurePoolName << " from pool must be text.");

            columnIndexesReorderMap->insert(
                {textFeature.Position.FlatIndex, textFeature.Position.FlatIndex});
        }
    }

    void CheckModelAndDatasetCompatibility(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData)
    {
        THashMap<ui32, ui32> columnReorderMap;
        CheckModelAndDatasetCompatibility(model, objectsData, &columnReorderMap);
    }

    TVector<TVector<ui8>> GetFloatFeaturesBordersRemap(
        const TFullModel& model,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo)
    {
        TVector<TVector<ui8>> floatBinsRemap(model.ModelTrees->GetFloatFeatures().size());
        for (const auto& feature: model.ModelTrees->GetFloatFeatures()) {
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
