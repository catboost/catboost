#include "approx_dimension.h"

#include <catboost/private/libs/labels/label_converter.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/metric_options.h>


namespace NCB {

    ui32 GetApproxDimension(
        const NCatboostOptions::TCatBoostOptions& catBoostOptions,
        const TLabelConverter& labelConverter,
        ui32 targetDimension
    ) {
        const ELossFunction lossFunction = catBoostOptions.LossFunctionDescription.Get().GetLossFunction();
        if (lossFunction == ELossFunction::RMSEWithUncertainty) {
            return ui32(2);
        } else if (lossFunction == ELossFunction::MultiQuantile) {
            const auto& paramsMap = catBoostOptions.LossFunctionDescription.Get().GetLossParams().GetParamsMap();
            return NCatboostOptions::GetAlphaMultiQuantile(paramsMap).size();
        } else if (lossFunction == ELossFunction::SurvivalAft) {
            return ui32(1);
        } else if (IsMultiTargetObjective(lossFunction)) {
            return targetDimension;
        } else {
            if (labelConverter.IsInitialized()) {
                return (ui32)labelConverter.GetApproxDimension();
            }
            return ui32(1);
        }
    }

}
