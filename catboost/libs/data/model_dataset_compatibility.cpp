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

static inline TString GetFeatureName(const TString& featureId, int featureIndex) {
    return featureId == "" ? ToString(featureIndex) : featureId;
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

template <class TFeature>
static void CheckFeatureTypesAndNames(
    TConstArrayRef<TFeature> modelFeatures,
    const THashSet<ui32>& datasetFeaturesIndices,
    TConstArrayRef<NCB::TFeatureMetaInfo> datasetFeaturesMetaInfo,
    THashMap<ui32, ui32>* columnIndexesReorderMap
) {
    for (const TFeature& feature : modelFeatures) {
        if (!feature.UsedInModel()) {
            continue;
        }
        TString featureModelName = GetFeatureName(feature.FeatureId, feature.Position.FlatIndex);
        CB_ENSURE(
            SafeIntegerCast<size_t>(feature.Position.FlatIndex) < datasetFeaturesMetaInfo.size(),
            "Feature " << featureModelName << " is present in model but not in pool.");
        if (SafeIntegerCast<size_t>(feature.Position.FlatIndex) < datasetFeaturesMetaInfo.size()
                && feature.FeatureId != ""
                && datasetFeaturesMetaInfo[feature.Position.FlatIndex].Name != "")
        {
            CB_ENSURE(
                feature.FeatureId == datasetFeaturesMetaInfo[feature.Position.FlatIndex].Name,
                "At position " << feature.Position.FlatIndex
                << " should be feature with name " << feature.FeatureId
                << " (found " << datasetFeaturesMetaInfo[feature.Position.FlatIndex].Name << ").");
        }
        TString featurePoolName;
        if (SafeIntegerCast<size_t>(feature.Position.FlatIndex) < datasetFeaturesMetaInfo.size()) {
            featurePoolName = GetFeatureName(
                datasetFeaturesMetaInfo[feature.Position.FlatIndex].Name,
                feature.Position.FlatIndex);
        } else {
            featurePoolName = GetFeatureName("", feature.Position.FlatIndex);
        }
        CB_ENSURE(
            datasetFeaturesIndices.contains(feature.Position.FlatIndex),
            "Feature " << featurePoolName << " from pool must be " << FeatureTypeName<TFeature>() << ".");

        columnIndexesReorderMap->insert(
            {feature.Position.FlatIndex, feature.Position.FlatIndex});
    }
}

namespace NCB {

    bool CheckColumnRemappingPossible(
        const TFullModel& model,
        const TFeaturesLayout& datasetFeaturesLayout,
        const THashSet<ui32>& datasetFloatFeatureFlatIndexes,
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

        CheckFeatureTypes(
            model.ModelTrees->GetFloatFeatures(),
            datasetFeatureNamesMap,
            datasetFloatFeatureFlatIndexes,
            columnIndexesReorderMap
        );
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

        const auto datasetFloatFeatureInternalIdxToExternalIdx =
            datasetFeaturesLayout.GetFloatFeatureInternalIdxToExternalIdx();

        THashSet<ui32> datasetFloatFeatures(
            datasetFloatFeatureInternalIdxToExternalIdx.begin(),
            datasetFloatFeatureInternalIdxToExternalIdx.end());

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
            datasetFloatFeatures,
            datasetCatFeatures,
            datasetTextFeatures,
            columnIndexesReorderMap))
        {
            return;
        }

        columnIndexesReorderMap->clear();

        const auto& datasetFeaturesMetaInfo = datasetFeaturesLayout.GetExternalFeaturesMetaInfo();

        CheckFeatureTypesAndNames(
            model.ModelTrees->GetFloatFeatures(),
            datasetFloatFeatures,
            datasetFeaturesMetaInfo,
            columnIndexesReorderMap
        );

        CheckFeatureTypesAndNames(
            model.ModelTrees->GetCatFeatures(),
            datasetCatFeatures,
            datasetFeaturesMetaInfo,
            columnIndexesReorderMap
        );

        CheckFeatureTypesAndNames(
            model.ModelTrees->GetTextFeatures(),
            datasetTextFeatures,
            datasetFeaturesMetaInfo,
            columnIndexesReorderMap
        );
    }

    void CheckModelAndDatasetCompatibility(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData)
    {
        THashMap<ui32, ui32> columnReorderMap;
        CheckModelAndDatasetCompatibility(model, objectsData, &columnReorderMap);
    }

    TVector<ui8> GetFloatFeatureBordersRemap(
        const TFloatFeature& feature,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo) {
        CB_ENSURE(
            !feature.Borders.empty(),
            "Feature " << feature.Position.FlatIndex <<  ": model does not have border information for it"
        );
        CB_ENSURE(
            quantizedFeaturesInfo.HasBorders(NCB::TFloatFeatureIdx(feature.Position.FlatIndex)),
            "Feature " << feature.Position.FlatIndex <<  ": dataset does not have border information for it"
        );

        TVector<ui8> floatBinsRemap;
        auto& quantizedBorders = quantizedFeaturesInfo.GetBorders(NCB::TFloatFeatureIdx(feature.Position.FlatIndex));
        ui32 poolBucketIdx = 0;
        auto addRemapBinIdx = [&] (ui8 bucketIdx) {
            floatBinsRemap.push_back(bucketIdx);
            ++poolBucketIdx;
        };
        for (ui32 modelBucketIdx = 0; modelBucketIdx < feature.Borders.size(); ++modelBucketIdx) {
            while (poolBucketIdx < quantizedBorders.size() &&
                quantizedBorders[poolBucketIdx] < feature.Borders[modelBucketIdx]) {
                addRemapBinIdx(modelBucketIdx);
            }
            CB_ENSURE(
                poolBucketIdx < quantizedBorders.size(),
                "Feature " << feature.Position.FlatIndex << ": inconsistent borders, last quantized vs model: "
                << double(quantizedBorders.back()) << " vs " << feature.Borders[modelBucketIdx]
            );
            CB_ENSURE(
                quantizedBorders[poolBucketIdx] == feature.Borders[modelBucketIdx],
                "Feature " << feature.Position.FlatIndex << ": inconsistent borders, quantized vs model: "
                << double(quantizedBorders[poolBucketIdx]) << " vs " << feature.Borders[modelBucketIdx]
            );
            addRemapBinIdx(modelBucketIdx);
        }
        while (poolBucketIdx <= quantizedBorders.size()) {
            addRemapBinIdx(feature.Borders.size());
        }
        return floatBinsRemap;
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
            floatBinsRemap[feature.Position.FlatIndex] =
                GetFloatFeatureBordersRemap(feature, quantizedFeaturesInfo);
        }
        return floatBinsRemap;
    }
}
