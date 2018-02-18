#pragma once

#include "enums.h"
#include "option.h"
#include "json_helper.h"
#include "loss_description.h"

namespace NCatboostOptions {
    struct TMetricOptions {
    public:
        explicit TMetricOptions()
            : EvalMetric("eval_metric", TLossDescription())
            , CustomMetrics("custom_metrics", TVector<TLossDescription>()) {
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(options, &EvalMetric, &CustomMetrics);
            CB_ENSURE(EvalMetric.Get().GetLossFunction() != ELossFunction::CtrFactor, ToString(ELossFunction::CtrFactor) << " cannot be used for overfitting detection or selecting best iteration on validation");
        }

        void Save(NJson::TJsonValue* options) const {
            SaveFields(options, EvalMetric, CustomMetrics);
        }

        bool operator==(const TMetricOptions& rhs) const {
            return std::tie(EvalMetric, CustomMetrics) == std::tie(rhs.EvalMetric, rhs.CustomMetrics);
        }

        bool operator!=(const TMetricOptions& rhs) const {
            return !(rhs == *this);
        }

        TOption<TLossDescription> EvalMetric;
        TOption<TVector<TLossDescription>> CustomMetrics;
    };
}
