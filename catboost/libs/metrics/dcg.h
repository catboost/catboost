#pragma once

#include <util/generic/fwd.h>
#include <util/generic/maybe.h>

// TODO(yazevnul): add fwd header for NMetrics
namespace NMetrics {
    struct TSample;
}

double CalcNDCG(TConstArrayRef<NMetrics::TSample> samples);

double CalcDCG(TConstArrayRef<NMetrics::TSample> samples, TMaybe<double> expDecay = {});
double CalcIDCG(TConstArrayRef<NMetrics::TSample> samples, TMaybe<double> expDecay = {});
