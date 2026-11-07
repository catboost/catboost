#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/labels/external_label_helper.h>
#include <catboost/private/libs/target/classification_target_helper.h>

namespace NCB {
    // CatBoost compresses observed multiclass labels for training. Baselines
    // retain their public class order, and the model metadata restores missing
    // classes for public predictions. Reuse that mapping in the Metal cursor.
    inline TVector<ui32> GetMetalBaselineColumns(
        const TClassificationTargetHelper& classificationTargetHelper,
        ui32 approxDimension, const TFullModel* initModel)
    {
        TVector<ui32> result(approxDimension);
        if (!classificationTargetHelper.IsInitialized() || !classificationTargetHelper.IsMultiClass()) {
            for (ui32 dimension = 0; dimension < approxDimension; ++dimension) result[dimension] = dimension;
            return result;
        }
        TFullModel classModel;
        classModel.ModelTrees.GetMutable()->SetApproxDimension(approxDimension);
        classModel.ModelInfo["class_params"] = classificationTargetHelper.Serialize();
        const TExternalLabelsHelper labels(classModel);
        for (ui32 dimension = 0; dimension < approxDimension; ++dimension) {
            result[dimension] = labels.GetExternalIndex(dimension);
        }
        if (initModel) {
            CB_ENSURE(initModel->GetDimensionsCount() == approxDimension &&
                      initModel->GetModelClassLabels() == classModel.GetModelClassLabels(),
                      "Metal initial model class labels differ from the current training labels");
            const TExternalLabelsHelper initialLabels(*initModel);
            CB_ENSURE(initialLabels.GetExternalApproxDimension() == labels.GetExternalApproxDimension(),
                      "Metal initial model class count differs from the current training labels");
            for (ui32 dimension = 0; dimension < approxDimension; ++dimension) {
                CB_ENSURE(initialLabels.GetExternalIndex(dimension) == static_cast<int>(result[dimension]),
                          "Metal initial model class order differs from the current training labels");
            }
        }
        return result;
    }
}
