#pragma once

#include "shap_values.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/system/types.h>


TVector<TVector<double>> CalcSageValues(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    int logPeriod,
    NPar::ILocalExecutor* localExecutor,
    size_t nSamples = 128,
    size_t batchSize = 512,
    bool detectConvergence = true
);

void CalcAndOutputSageValues(
    const TFullModel& model,
    const NCB::TDataProvider& dataset,
    int logPeriod,
    const TString& outputPath,
    NPar::ILocalExecutor* localExecutor,
    size_t nSamples = 128,
    size_t batchSize = 512,
    bool detectConvergence = true
);
