#pragma once

#include <catboost/libs/model/model.h>

/// builder for both TExternalLabelHelper and TLabelConverter
template <typename TLabelsHelper>
TLabelsHelper BuildLabelsHelper(const TFullModel& model) {
    TLabelsHelper labelsHelper;
    if (model.ObliviousTrees.ApproxDimension > 1) {  // is multiclass?
        if (model.ModelInfo.has("multiclass_params")) {
            labelsHelper.Initialize(model.ModelInfo.at("multiclass_params"));
        }
        else {
            labelsHelper.Initialize(model.ObliviousTrees.ApproxDimension);
        }
    }
    return labelsHelper;
}

