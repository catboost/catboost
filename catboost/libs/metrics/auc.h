#pragma once

#include "sample.h"

double CalcAUC(TVector<NMetrics::TSample>* samples, double* outWeightSum = nullptr, double* outPairWeightSum = nullptr);
double CalcGINI(TVector<NMetrics::TSample>* samples, double* outWeightSum = nullptr, double* outPairWeightSum = nullptr);