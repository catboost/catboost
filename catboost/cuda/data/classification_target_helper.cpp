#include "classification_target_helper.h"
#include <catboost/libs/helpers/binarize_target.h>
#include <catboost/libs/options/loss_description.h>



TString NCatboostCuda::TClassificationTargetHelper::Serialize() const {
    if (LabelConverter.IsInitialized()) {
        return LabelConverter.SerializeMulticlassParams((int)Options.DataProcessingOptions->ClassesCount.Get(), Options.DataProcessingOptions->ClassNames);
    }
    return "";
}

void NCatboostCuda::TClassificationTargetHelper::MakeTargetAndWeights(bool isLearnTarget,
                                                                      TVector<float>* loadedTargets,
                                                                      TVector<float>* loadedWeights) {

    const auto isMultiClass = IsMultiClassMetric(Options.LossFunctionDescription->GetLossFunction());

    if (isMultiClass) {
        if (!LabelConverter.IsInitialized()) {
            CB_ENSURE(isLearnTarget);
            LabelConverter.Initialize(*loadedTargets, GetClassesCount(Options.DataProcessingOptions->ClassesCount,
                                                                      Options.DataProcessingOptions->ClassNames));
        }
        CB_ENSURE(LabelConverter.IsInitialized());

    }

    if (Options.LossFunctionDescription->GetLossFunction() == ELossFunction::Logloss) {
        PrepareTargetBinary(NCatboostOptions::GetLogLossBorder(Options.LossFunctionDescription), loadedTargets);
    }

    auto& classWeights = Options.DataProcessingOptions->ClassWeights.Get();
    auto& targets = *loadedTargets;
    auto& weights = *loadedWeights;
    Y_VERIFY(targets.size() == weights.size());

    if (!classWeights.empty()) {
        for (size_t i = 0; i < targets.size(); ++i) {
            const ui32 clazz = static_cast<ui32>(targets[i]);
            CB_ENSURE(clazz < classWeights.size(), "class #" << clazz << " is missing in class weights");
            weights[i] *= classWeights[clazz];
        }
    }

    if (isMultiClass) {
        PrepareTargetCompressed(LabelConverter, loadedTargets);
    }
}
