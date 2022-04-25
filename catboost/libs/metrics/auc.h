#pragma once

#include "sample.h"

#include <library/cpp/threading/local_executor/local_executor.h>

double CalcAUC(
    TVector<NMetrics::TSample>* samples,
    double* outWeightSum = nullptr,
    double* outPairWeightSum = nullptr,
    NPar::ILocalExecutor* localExecutor = nullptr);

double CalcBinClassAuc(TVector<NMetrics::TBinClassSample>* positiveSamples, TVector<NMetrics::TBinClassSample>* negativeSamples, NPar::ILocalExecutor* localExecutor);
double CalcBinClassAuc(TVector<NMetrics::TBinClassSample>* positiveSamples, TVector<NMetrics::TBinClassSample>* negativeSamples, int threadCount = 1);
