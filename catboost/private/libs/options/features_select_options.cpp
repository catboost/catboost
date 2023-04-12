#include "features_select_options.h"

#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/options/json_helper.h>


NCatboostOptions::TFeaturesSelectOptions::TFeaturesSelectOptions()
    : FeaturesForSelect("features_for_select", TVector<ui32>())
    , NumberOfFeaturesToSelect("num_features_to_select", 1)
    , FeaturesTagsForSelect("features_tags_for_select", TVector<TString>())
    , NumberOfFeaturesTagsToSelect("num_features_tags_to_select", 1)
    , Steps("features_selection_steps", 1)
    , TrainFinalModel("train_final_model", false)
    , ResultPath("features_selection_result_path", "selection_result.json")
    , Algorithm("features_selection_algorithm", NCB::EFeaturesSelectionAlgorithm::RecursiveByShapValues)
    , Grouping("features_selection_grouping", NCB::EFeaturesSelectionGrouping::Individual)
    , ShapCalcType("shap_calc_type", ECalcTypeShapValues::Regular)
{
}

bool NCatboostOptions::TFeaturesSelectOptions::operator==(const TFeaturesSelectOptions& rhs) const {
    const auto& options = std::tie(
        FeaturesForSelect,
        NumberOfFeaturesToSelect,
        FeaturesTagsForSelect,
        NumberOfFeaturesTagsToSelect,
        Steps,
        TrainFinalModel,
        ResultPath,
        Algorithm,
        Grouping,
        ShapCalcType);
    const auto& rhsOptions = std::tie(
        rhs.FeaturesForSelect,
        rhs.NumberOfFeaturesToSelect,
        rhs.FeaturesTagsForSelect,
        rhs.NumberOfFeaturesTagsToSelect,
        rhs.Steps,
        rhs.TrainFinalModel,
        rhs.ResultPath,
        rhs.Algorithm,
        rhs.Grouping,
        rhs.ShapCalcType);
    return options == rhsOptions;
}

bool NCatboostOptions::TFeaturesSelectOptions::operator!=(const TFeaturesSelectOptions& rhs) const {
    return !(rhs == *this);
}

void NCatboostOptions::TFeaturesSelectOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(
        options,
        &FeaturesForSelect,
        &NumberOfFeaturesToSelect,
        &FeaturesTagsForSelect,
        &NumberOfFeaturesTagsToSelect,
        &Steps,
        &TrainFinalModel,
        &ResultPath,
        &Algorithm,
        &Grouping,
        &ShapCalcType);
}

void NCatboostOptions::TFeaturesSelectOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(
        options,
        FeaturesForSelect,
        NumberOfFeaturesToSelect,
        FeaturesTagsForSelect,
        NumberOfFeaturesTagsToSelect,
        Steps,
        TrainFinalModel,
        ResultPath,
        Algorithm,
        Grouping,
        ShapCalcType);
}

void NCatboostOptions::TFeaturesSelectOptions::CheckAndUpdateSteps() {
    auto adjustSteps = [&] (int subsetSize, int toSelectCount, TStringBuf groupingName) {
        const auto eliminateCount = subsetSize - toSelectCount;
        if (Steps > eliminateCount) {
            CATBOOST_WARNING_LOG << "The number of " << groupingName << " selection steps (" << Steps
                << ") is greater than the number of " << groupingName << " to eliminate (" << eliminateCount
                << "). The number of steps was reduced to " << eliminateCount << "." << Endl;
            Steps = eliminateCount;
        }
    };

    if (Grouping == NCB::EFeaturesSelectionGrouping::Individual) {
        adjustSteps((int)FeaturesForSelect->size(), NumberOfFeaturesToSelect, "features");
    } else { // ByTag
        adjustSteps((int)FeaturesTagsForSelect->size(), NumberOfFeaturesTagsToSelect, "features tags");
    }
}

