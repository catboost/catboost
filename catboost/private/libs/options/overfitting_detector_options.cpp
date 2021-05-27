#include "overfitting_detector_options.h"
#include "json_helper.h"

#include <catboost/libs/helpers/exception.h>

#include <library/cpp/json/json_value.h>

NCatboostOptions::TOverfittingDetectorOptions::TOverfittingDetectorOptions()
    : AutoStopPValue("stop_pvalue", 0)
    , OverfittingDetectorType("type", EOverfittingDetectorType::IncToDec)
    , IterationsWait("wait_iterations", 20)
{
}

bool NCatboostOptions::TOverfittingDetectorOptions::operator==(const TOverfittingDetectorOptions& rhs) const {
    return std::tie(AutoStopPValue, OverfittingDetectorType, IterationsWait) ==
        std::tie(rhs.AutoStopPValue, rhs.OverfittingDetectorType, rhs.IterationsWait);
}

bool NCatboostOptions::TOverfittingDetectorOptions::operator!=(const TOverfittingDetectorOptions& rhs) const {
    return !(rhs == *this);
}

void NCatboostOptions::TOverfittingDetectorOptions::Load(const NJson::TJsonValue& options) {
    if (!options.Has("type")) {
        if (options.Has("stop_pvalue")) {
            OverfittingDetectorType.Set(EOverfittingDetectorType::IncToDec);
        } else if (options.Has("wait_iterations")) {
            OverfittingDetectorType.Set(EOverfittingDetectorType::Iter);
        } else {
            OverfittingDetectorType.Set(EOverfittingDetectorType::None);
        }
    }
    CheckedLoad(options, &AutoStopPValue, &OverfittingDetectorType, &IterationsWait);
    CB_ENSURE(
            (OverfittingDetectorType.Get() != EOverfittingDetectorType::Iter)
            || !options.Has("stop_pvalue")
            || (options["stop_pvalue"].GetDouble() == 0.0),  // this check is needed because the default value is serialized
            "Auto-stop PValue is not a valid parameter for Iter overfitting detector."
            );
    Validate();
}

void NCatboostOptions::TOverfittingDetectorOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, AutoStopPValue, OverfittingDetectorType, IterationsWait);
}

void NCatboostOptions::TOverfittingDetectorOptions::Validate() const {
    CB_ENSURE(IterationsWait.Get() > 0, "Wait iterations in OD-detector should be > 0");
    CB_ENSURE(AutoStopPValue.Get() >= 0, "Auto-stop PValue in OD-detector should be >= 0");
}
