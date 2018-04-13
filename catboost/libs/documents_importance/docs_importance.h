#pragma once

#include "docs_importance_helpers.h"

struct TDStrResult {
    TDStrResult() = default;
    TDStrResult(ui32 testDocCount, ui32 topSize)
        : Indices(testDocCount)
        , Scores(testDocCount, TVector<double>(topSize))
    {
    }

    TVector<TVector<ui32>> Indices; // [TestDocDount][Min(TopSize, TrainDocCount)]
    TVector<TVector<double>> Scores; // [TestDocDount][Min(TopSize, TrainDocCount)]
};

TDStrResult GetDocumentImportances(
    const TFullModel& model,
    const TPool& trainPool,
    const TPool& testPool,
    const TString& dstrType,
    int topSize,
    const TString& updateMethod,
    int threadCount
);

