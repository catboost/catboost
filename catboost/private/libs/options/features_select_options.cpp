#include "features_select_options.h"

#include <catboost/private/libs/options/json_helper.h>


NCatboostOptions::TFeaturesSelectOptions::TFeaturesSelectOptions()
    : FeaturesForSelect("features_for_select", TVector<ui32>())
    , NumberOfFeaturesToSelect("num_features_to_select", 1)
    , Steps("features_selection_steps", 1)
    , TrainFinalModel("train_final_model", false)
    , ResultPath("features_selection_result_path", "selection_result.json")
    , Algorithm("features_selection_algorithm", NCB::EFeaturesSelectionAlgorithm::RecursiveByShapValues)
    , ShapCalcType("shap_calc_type", ECalcTypeShapValues::Regular)
{
}

void NCatboostOptions::TFeaturesSelectOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(
        options, &FeaturesForSelect, &NumberOfFeaturesToSelect, &Steps, &TrainFinalModel,
        &ResultPath, &Algorithm, &ShapCalcType
    );
}

void NCatboostOptions::TFeaturesSelectOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(
        options, FeaturesForSelect, NumberOfFeaturesToSelect, Steps, TrainFinalModel,
        ResultPath, Algorithm, ShapCalcType
    );
}

bool NCatboostOptions::TFeaturesSelectOptions::operator==(const TFeaturesSelectOptions& rhs) const {
    const auto& options = std::tie(FeaturesForSelect, NumberOfFeaturesToSelect, Steps,
                                   TrainFinalModel, ResultPath, Algorithm, ShapCalcType);
    const auto& rhsOptions = std::tie(rhs.FeaturesForSelect, rhs.NumberOfFeaturesToSelect, rhs.Steps,
                                      rhs.TrainFinalModel, rhs.ResultPath, rhs.Algorithm, rhs.ShapCalcType);
    return options == rhsOptions;
}

bool NCatboostOptions::TFeaturesSelectOptions::operator!=(const TFeaturesSelectOptions& rhs) const {
    return !(rhs == *this);
}
