#pragma once

#include "sample.h"

#include <library/threading/local_executor/local_executor.h>

double CalcAUC(TVector<NMetrics::TSample>* samples, NPar::TLocalExecutor* localExecutor, double* outWeightSum = nullptr, double* outPairWeightSum = nullptr);
double CalcAUC(TVector<NMetrics::TSample>* samples, double* outWeightSum = nullptr, double* outPairWeightSum = nullptr, int threadCount = 1);
