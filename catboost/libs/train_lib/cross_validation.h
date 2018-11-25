#pragma once

#include <catboost/libs/data/pool.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/options/cross_validation_params.h>

#include <library/json/json_value.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>


struct TCVIterationResults {
    double AverageTrain;
    double StdDevTrain;
    double AverageTest;
    double StdDevTest;
};

struct TCVResult {
    TString Metric;
    TVector<double> AverageTrain;
    TVector<double> StdDevTrain;
    TVector<double> AverageTest;
    TVector<double> StdDevTest;

    void AppendOneIterationResults(const TCVIterationResults& results) {
        AverageTrain.push_back(results.AverageTrain);
        StdDevTrain.push_back(results.StdDevTrain);
        AverageTest.push_back(results.AverageTest);
        StdDevTest.push_back(results.StdDevTest);
    }
};

void CrossValidate(
    const NJson::TJsonValue& plainJsonParams,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TPool& pool,
    const TCrossValidationParams& cvParams,
    TVector<TCVResult>* results);
