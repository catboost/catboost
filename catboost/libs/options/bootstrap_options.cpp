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
            default: {
                if (BaggingTemperature.IsSet()) {
                    ythrow TCatboostException() << "Error: bagging temperature available for bayesian bootstrap only";
                }
                if (TakenFraction.Get() >= 0.99f) {
                    MATRIXNET_WARNING_LOG << "Big taken fraction (" << TakenFraction.Get() << ") dissables bagging. If you don't want bootstrap, just set bootstrap-type No.";
                }
                break;
            }
        }
    }
}
