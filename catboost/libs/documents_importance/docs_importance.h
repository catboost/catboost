#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>

#include <util/generic/string.h>
#include <util/system/types.h>


struct TDStrResult {
    TDStrResult() = default;
    TDStrResult(ui32 testDocCount)
        : Indices(testDocCount)
        , Scores(testDocCount)
    {
    }

    TVector<TVector<ui32>> Indices; // [TestDocDount][Min(TopSize, TrainDocCount)]
    TVector<TVector<double>> Scores; // [TestDocDount][Min(TopSize, TrainDocCount)]
};

TDStrResult GetDocumentImportances(
    const TFullModel& model,
    const NCB::TDataProvider& trainData,
    const NCB::TDataProvider& testData,
    const TString& dstrTypeStr,
    int topSize,
    const TString& updateMethodStr,
    const TString& importanceValuesSignStr,
    int threadCount,
    int logPeriod = 0
);

