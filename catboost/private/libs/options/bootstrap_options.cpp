#include "bootstrap_options.h"

namespace NCatboostOptions {
    void TBootstrapConfig::Validate() const {
        CB_ENSURE((GetTakenFraction() > 0) && (GetTakenFraction() <= 1.0f), "Subsample should be in (0,1]");
        CB_ENSURE(GetBaggingTemperature() >= 0, "Bagging temperature should be >= 0");
        CB_ENSURE(GetMvsReg().OrElse(0) >= 0, "MVS regularization parameter should be >= 0");

        if (BootstrapType.NotSet()) {
            return;
        }
        EBootstrapType type = BootstrapType;
        switch (type) {
            case EBootstrapType::Bayesian: {
                if (TakenFraction.IsSet()) {
                    ythrow TCatBoostException()
                        << "Error: bayesian bootstrap doesn't support 'subsample' option";
                }
                break;
            }
            case EBootstrapType::No: {
                if (BaggingTemperature.IsSet() || TakenFraction.IsSet()) {
                    ythrow TCatBoostException() << "Error: you shoudn't provide bootstrap options if bootstrap is disabled";
                }
                break;
            }
            case EBootstrapType::Poisson: {
                if (TaskType == ETaskType::CPU) {
                    ythrow TCatBoostException()
                        << "Error: poisson bootstrap is not supported on CPU";
                }
                break;
            }
            case EBootstrapType::MVS: {
                CB_ENSURE(
                    GetSamplingUnit() == ESamplingUnit::Object,
                    "MVS bootstrap supports per object sampling only."
                );
                break;
            }
            default: {
                Y_ASSERT(type == EBootstrapType::Bernoulli);
                if (BaggingTemperature.IsSet()) {
                    ythrow TCatBoostException() << "Error: bagging temperature available for bayesian bootstrap only";
                }
                break;
            }
        }
    }

}
