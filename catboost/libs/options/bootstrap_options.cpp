#include "bootstrap_options.h"

namespace NCatboostOptions {
    void TBootstrapConfig::Validate() const {
        CB_ENSURE((GetTakenFraction() > 0) && (GetTakenFraction() <= 1.0f), "Taken fraction should in in (0,1]");
        CB_ENSURE(GetBaggingTemperature() >= 0, "Bagging temperature should be >= 0");
        CB_ENSURE((GetMvsHeadFraction() > 0) && (GetMvsHeadFraction() <= 1.0f), "MVS head fraction should be in (0,1]");

        EBootstrapType type = BootstrapType;
        switch (type) {
            case EBootstrapType::Bayesian: {
                if (TakenFraction.IsSet()) {
                    ythrow TCatBoostException()
                        << "Error: bayesian bootstrap doesn't support taken fraction option";
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
                if (TaskType != ETaskType::CPU) {
                    ythrow TCatBoostException()
                        << "Error: MVS bootstrap is supported only on CPU";
                }
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
