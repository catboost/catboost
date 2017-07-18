#pragma once

#include "sample.h"

double CalcAUC(yvector<NMetrics::TSample>* samples, double* outWeightSum = nullptr, double* outPairWeightSum = nullptr);
