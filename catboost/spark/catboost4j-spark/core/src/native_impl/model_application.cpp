#include "model_application.h"

#include <catboost/private/libs/algo/apply.h>

#include <catboost/libs/data/features_layout_helpers.h>
#include <catboost/libs/data/model_dataset_compatibility.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/ymath.h>

using namespace NCB;


TApplyResultIterator::TApplyResultIterator(
    const TFullModel& model,
    NCB::TObjectsDataProviderPtr objectsDataProvider,
    EPredictionType predictionType,
    NPar::TLocalExecutor* localExecutor
)
    : ApplyResult(
          ApplyModelMulti(
              model,
              *objectsDataProvider,
              predictionType,
              /*begin*/ 0,
              /*end*/ 0,
              localExecutor
          )
      )
{}


static void CheckIfModelBordersAreSubsetOfDatasetBorders(
    TConstArrayRef<float> bordersInModel,
    TConstArrayRef<float> bordersInDataset,
    TStringBuf featureName
) {
    auto datasetBorderIterator = bordersInDataset.begin();
    float borderInDataset = 0.0f;
    for (auto borderInModel : bordersInModel) {
        do {
            CB_ENSURE(
                datasetBorderIterator != bordersInDataset.end(),
                "border with value " << borderInModel << " for feature '" << featureName
                << "' is present in the model but not found in the dataset borders"
            );
            borderInDataset = *datasetBorderIterator++;
        } while (!FuzzyEquals(borderInModel, borderInDataset));
    }
}

static void CheckIfDatasetNanModeIsCompatibleWithModelNanValueTreatment(
    ENanMode datasetNanMode,
    TFloatFeature::ENanValueTreatment modelNanValueTreatment,
    TStringBuf featureName
) {
    auto areCompatible = [&] () {
        switch (modelNanValueTreatment) {
            case TFloatFeature::ENanValueTreatment::AsIs:
            case TFloatFeature::ENanValueTreatment::AsFalse:
                return datasetNanMode != ENanMode::Max;
            case TFloatFeature::ENanValueTreatment::AsTrue:
                return datasetNanMode != ENanMode::Min;
        }
    };

    CB_ENSURE(
        areCompatible(),
        "Feature '" << featureName << "' : Model has nanValueTreatment="
        << modelNanValueTreatment << ", but dataset has nanMode=" << datasetNanMode
    );
}


void CheckModelAndDatasetCompatibility(
    const TFullModel& model,
    const NCB::TQuantizedFeaturesInfo& datasetQuantizedFeaturesInfo
) {
    const auto& datasetFeaturesLayout = *datasetQuantizedFeaturesInfo.GetFeaturesLayout();

    THashMap<ui32, ui32> columnIndexesReorderMap;
    NCB::CheckModelAndDatasetCompatibility(model, datasetFeaturesLayout, &columnIndexesReorderMap);

    for (const auto& floatFeature : model.ModelTrees->GetFloatFeatures()) {
        if (!floatFeature.UsedInModel()) {
            continue;
        }
        auto datasetFlatFeatureIdx = columnIndexesReorderMap.at(floatFeature.Position.FlatIndex);
        auto datasetFloatFeatureIdx = datasetFeaturesLayout.GetInternalFeatureIdx<EFeatureType::Float>(
            datasetFlatFeatureIdx
        );
        CheckIfModelBordersAreSubsetOfDatasetBorders(
            floatFeature.Borders,
            datasetQuantizedFeaturesInfo.GetBorders(datasetFloatFeatureIdx),
            floatFeature.FeatureId
        );
        CheckIfDatasetNanModeIsCompatibleWithModelNanValueTreatment(
            datasetQuantizedFeaturesInfo.GetNanMode(datasetFloatFeatureIdx),
            floatFeature.NanValueTreatment,
            floatFeature.FeatureId
        );
    }
}

TQuantizedFeaturesInfoPtr CreateQuantizedFeaturesInfoForModelApplication(
    const TFullModel& model,
    const TFeaturesLayout& datasetFeaturesLayout
) {
    THashMap<ui32, ui32> columnIndexesReorderMap;
    NCB::CheckModelAndDatasetCompatibility(model, datasetFeaturesLayout, &columnIndexesReorderMap);

    auto datasetFeaturesMetaInfo = datasetFeaturesLayout.GetExternalFeaturesMetaInfo();
    TVector<TFeatureMetaInfo> quantizedDatasetFeaturesMetaInfo(
        datasetFeaturesMetaInfo.begin(),
        datasetFeaturesMetaInfo.end()
    );

    for (auto& featureMetaInfo : quantizedDatasetFeaturesMetaInfo) {
        featureMetaInfo.IsIgnored = true;
        featureMetaInfo.IsAvailable = false;
    }

    for (const auto& [modelFlatFeatureIdx, dataFlatFeatureIdx] : columnIndexesReorderMap) {
        quantizedDatasetFeaturesMetaInfo[dataFlatFeatureIdx].IsIgnored = false;
        quantizedDatasetFeaturesMetaInfo[dataFlatFeatureIdx].IsAvailable = true;
    }

    auto quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
        TFeaturesLayout(&quantizedDatasetFeaturesMetaInfo),
        /*ignoredFeatures*/ TConstArrayRef<ui32>(),
        NCatboostOptions::TBinarizationOptions()
    );

    for (const auto& floatFeature : model.ModelTrees->GetFloatFeatures()) {
        if (!floatFeature.UsedInModel()) {
            continue;
        }
        auto datasetFlatFeatureIdx = columnIndexesReorderMap.at(floatFeature.Position.FlatIndex);
        auto datasetFloatFeatureIdx = datasetFeaturesLayout.GetInternalFeatureIdx<EFeatureType::Float>(
            datasetFlatFeatureIdx
        );
        quantizedFeaturesInfo->SetBorders(datasetFloatFeatureIdx, TVector<float>(floatFeature.Borders));
        quantizedFeaturesInfo->SetNanMode(
            datasetFloatFeatureIdx,
            (floatFeature.NanValueTreatment == TFloatFeature::ENanValueTreatment::AsTrue) ?
                ENanMode::Max
                : ENanMode::Min
        );
    }

    return quantizedFeaturesInfo;
}

