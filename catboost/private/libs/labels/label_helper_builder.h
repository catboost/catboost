#pragma once

#include <catboost/libs/model/model.h>

/// builder for both TExternalLabelHelper and TLabelConverter
template <typename TLabelsHelper>
TLabelsHelper BuildLabelsHelper(const TFullModel& model) {
    TLabelsHelper labelsHelper;
    if (model.GetDimensionsCount() > 1) {  // is multiclass?
        if (model.ModelInfo.contains("multiclass_params")) {
            labelsHelper.Initialize(model.ModelInfo.at("multiclass_params"));
        }
        else {
            labelsHelper.Initialize(model.GetDimensionsCount());
        }
    } else {
        const TVector<TString> binclassNames = model.GetModelClassNames();
        if (!binclassNames.empty()) {
            labelsHelper.Initialize(binclassNames);
        }
    }
    return labelsHelper;
}

