#pragma once

#include "enums.h"
#include "option.h"
#include "loss_description.h"

#include <util/generic/vector.h>

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    struct TMetricOptions {
    public:
        explicit TMetricOptions();

        void Save(NJson::TJsonValue* options) const;
        void Load(const NJson::TJsonValue& options);

        bool operator==(const TMetricOptions& rhs) const;
        bool operator!=(const TMetricOptions& rhs) const;

        TOption<TLossDescription> EvalMetric;
        TOption<TVector<TLossDescription>> CustomMetrics;
    };
}

bool IsMultiClass(ELossFunction lossFunction, const NCatboostOptions::TMetricOptions& metricOptions);
