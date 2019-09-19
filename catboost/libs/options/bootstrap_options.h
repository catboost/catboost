#pragma once

#include "enums.h"
#include "json_helper.h"
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <util/generic/yexception.h>
#include <cmath>

namespace NCatboostOptions {
    class TBootstrapConfig {
    public:
        explicit TBootstrapConfig(ETaskType taskType)
            : TakenFraction("subsample", 0.66f)
            , BaggingTemperature("bagging_temperature", 1.0)
            , MvsReg("mvs_reg", 1e-5, ETaskType::CPU)
            , BootstrapType("type", EBootstrapType::Bayesian)
            , SamplingUnit("sampling_unit", ESamplingUnit::Object)
            , TaskType(taskType)
        {
        }

        float GetTakenFraction() const {
            return TakenFraction.Get();
        }

        float GetPoissonLambda() const {
            const float takenFraction = TakenFraction.Get();
            return takenFraction < 1 ? -log(1 - takenFraction) : -1;
        }

        EBootstrapType GetBootstrapType() const {
            return BootstrapType.Get();
        }

        ESamplingUnit GetSamplingUnit() const {
            return SamplingUnit.Get();
        }

        float GetBaggingTemperature() const {
            return BaggingTemperature.Get();
        }

        float GetMvsReg() const {
            return MvsReg.Get();
        }

        bool MvsRegIsSet() const {
            return !MvsReg.NotSet();
        }

        void Validate() const;

        TOption<float>& GetTakenFraction() {
            return TakenFraction;
        }

        TOption<float>& GetBaggingTemperature() {
            return BaggingTemperature;
        }

        TOption<float>& GetMvsReg() {
            return MvsReg;
        }

        TOption<EBootstrapType>& GetBootstrapType() {
            return BootstrapType;
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(options, &TakenFraction, &BaggingTemperature, &MvsReg, &BootstrapType, &SamplingUnit);
        }

        void Save(NJson::TJsonValue* options) const {
            switch (BootstrapType.Get()) {
                case EBootstrapType::Bayesian: {
                    SaveFields(options, BaggingTemperature, BootstrapType);
                    break;
                }
                case EBootstrapType::No: {
                    SaveFields(options, BootstrapType);
                    break;
                }
                case EBootstrapType::MVS: {
                    SaveFields(options, MvsReg, BootstrapType);
                    break;
                }
                default: {
                    SaveFields(options, TakenFraction, BootstrapType);
                    break;
                }
            }
        }

        bool operator==(const TBootstrapConfig& rhs) const {
            return std::tie(TakenFraction, BaggingTemperature, MvsReg, BootstrapType) ==
                   std::tie(rhs.TakenFraction, rhs.BaggingTemperature, rhs.MvsReg, rhs.BootstrapType);
        }

        bool operator!=(const TBootstrapConfig& rhs) const {
            return !(rhs == *this);
        }


    private:
        TOption<float> TakenFraction;
        TOption<float> BaggingTemperature;
        TCpuOnlyOption<float> MvsReg;
        TOption<EBootstrapType> BootstrapType;
        TOption<ESamplingUnit> SamplingUnit;
        ETaskType TaskType;
    };

}
