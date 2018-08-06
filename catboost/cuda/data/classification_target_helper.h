#pragma once

#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/labels/label_converter.h>

namespace NCatboostCuda {

    class TClassificationTargetHelper {
    public:
        TClassificationTargetHelper(const NCatboostOptions::TCatBoostOptions& options)
        : Options(options){

        }

        void MakeTargetAndWeights(bool isLearnTarget, TVector<float>* loadedTargets, TVector<float>* loadedWeights);

        bool IsMultiClass() const {
            return IsMultiClassError(Options.LossFunctionDescription->GetLossFunction());
        }

        ui32 GetNumClasses() const {
            return LabelConverter.IsInitialized() ? LabelConverter.GetApproxDimension() : 2;
        }
        TString Serialize() const;

    private:
        TLabelConverter LabelConverter;
        const NCatboostOptions::TCatBoostOptions& Options;
    };
}
