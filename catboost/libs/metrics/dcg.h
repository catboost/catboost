#pragma once

#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <catboost/libs/options/enums.h>

// TODO(yazevnul): add fwd header for NMetrics
namespace NMetrics {
    struct TSample;
    enum class ENDCGMetricType;
}

double CalcNdcg(TConstArrayRef<NMetrics::TSample> samples, ENdcgMetricType type = ENdcgMetricType::Base);
double CalcDcg(
        TConstArrayRef<NMetrics::TSample> samplesRef,
        ENdcgMetricType type = ENdcgMetricType::Base,
        TMaybe<double> expDecay = Nothing());
double CalcIDcg(
        TConstArrayRef<NMetrics::TSample> samplesRef,
        ENdcgMetricType type = ENdcgMetricType::Base,
        TMaybe<double> expDecay = Nothing());
