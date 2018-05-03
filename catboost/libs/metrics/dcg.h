#pragma once

#include "sample.h"

#include <util/generic/algorithm.h>
#include <util/generic/maybe.h>

double CalcNDCG(TVector<NMetrics::TSample>* samples);

double CalcDCG(TVector<NMetrics::TSample>* samples, TMaybe<double> expDecay = Nothing());
double CalcIDCG(TVector<NMetrics::TSample>* samples, TMaybe<double> expDecay = Nothing());
