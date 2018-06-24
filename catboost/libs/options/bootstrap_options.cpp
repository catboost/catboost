#include "bootstrap_options.h"

namespace NCatboostOptions {
    void TBootstrapConfig::Validate() const {
        CB_ENSURE((GetTakenFraction() > 0) && (GetTakenFraction() <= 1.0f), "Taken fraction should in in (0,1]");
        CB_ENSURE(GetBaggingTemperature() >= 0, "Bagging temperature should be >= 0");

        EBootstrapType type = BootstrapType;
        switch (type) {
            case EBootstrapType::Bayesian: {
                if (TakenFraction.IsSet()) {
                    ythrow TCatboostException()
                        << "Error: bayesian bootstrap doesn't support taken fraction option";
                }
                break;
            }
            case EBootstrapType::No: {
                if (BaggingTemperature.IsSet() || TakenFraction.IsSet()) {
                    ythrow TCatboostException() << "Error: you shoudn't provide bootstrap options if bootstrap is disabled";
                }
                break;
            }
            case EBootstrapType::Poisson: {
                if (TaskType == ETaskType::CPU) {
                    ythrow TCatboostException()
                        << "Error: poisson bootstrap is not supported on CPU";
                }
                break;
            }
            default: {
                Y_ASSERT(type == EBootstrapType::Bernoulli);
                if (BaggingTemperature.IsSet()) {
                    ythrow TCatboostException() << "Error: bagging temperature available for bayesian bootstrap only";
                }
                break;
            }
        }
    }
}
