#pragma once

#include <catboost/private/libs/options/enums.h>

#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <util/generic/utility.h>

// TODO(yazevnul): add fwd header for NMetrics
namespace NMetrics {
    struct TSample;
}

double CalcDcgSorted(
    const TConstArrayRef<double> sortedTargets,
    TConstArrayRef<double> decay,
    const ENdcgMetricType type);

void FillDcgDecay(
    ENdcgDenominatorType denominator,
    TMaybe<double> expDecay,
    TArrayRef<double> decay);

double CalcDcg(
    TConstArrayRef<NMetrics::TSample> samples,
    TConstArrayRef<double> decay,
    ENdcgMetricType type = ENdcgMetricType::Base,
    ui32 topSize = Max<ui32>());

double CalcDcg(
    TConstArrayRef<NMetrics::TSample> samples,
    ENdcgMetricType type = ENdcgMetricType::Base,
    TMaybe<double> expDecay = Nothing(),
    ui32 topSize = Max<ui32>(),
    ENdcgDenominatorType denominator = ENdcgDenominatorType::LogPosition);

double CalcIDcg(
    TConstArrayRef<NMetrics::TSample> samples,
    TConstArrayRef<double> decay,
    ENdcgMetricType type = ENdcgMetricType::Base,
    ui32 topSize = Max<ui32>());

double CalcIDcg(
    TConstArrayRef<NMetrics::TSample> samples,
    ENdcgMetricType type = ENdcgMetricType::Base,
    TMaybe<double> expDecay = Nothing(),
    ui32 topSize = Max<ui32>(),
    ENdcgDenominatorType denominator = ENdcgDenominatorType::LogPosition);

double CalcNdcg(
    TConstArrayRef<NMetrics::TSample> samples,
    TConstArrayRef<double> decay,
    ENdcgMetricType type = ENdcgMetricType::Base,
    ui32 topSize = Max<ui32>());
