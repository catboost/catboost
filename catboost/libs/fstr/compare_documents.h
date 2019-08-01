#pragma once

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/options/enums.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/system/types.h>


TVector<TVector<double>> GetPredictionDiff(
    const TFullModel& model,
    const NCB::TDataProvider& dataProvider,
    NPar::TLocalExecutor* localExecutor
);

void CalcAndOutputPredictionDiff(
    const TFullModel& model,
    const NCB::TDataProvider& dataProvider,
    const TString& outputPath,
    NPar::TLocalExecutor* localExecutor
);
