#pragma once

#include <catboost/libs/algo/custom_objective_descriptor.h>
#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/options/cross_validation_params.h>

#include <library/json/json_value.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>

#include <numeric>


struct TCVIterationResults {
    double AverageTrain;
    double StdDevTrain;
    double AverageTest;
    double StdDevTest;
};

struct TCVResult {
    TString Metric;
    TVector<ui32> Iterations;
    TVector<double> AverageTrain;
    TVector<double> StdDevTrain;
    TVector<double> AverageTest;
    TVector<double> StdDevTest;

    void AppendOneIterationResults(ui32 iteration, const TCVIterationResults& results) {
        Iterations.push_back(iteration);
        AverageTrain.push_back(results.AverageTrain);
        StdDevTrain.push_back(results.StdDevTrain);
        AverageTest.push_back(results.AverageTest);
        StdDevTest.push_back(results.StdDevTest);
    }
};


TConstArrayRef<TString> GetTargetForStratifiedSplit(const NCB::TDataProvider& dataProvider);
TConstArrayRef<float> GetTargetForStratifiedSplit(const NCB::TTrainingDataProvider& dataProvider);

TVector<NCB::TArraySubsetIndexing<ui32>> CalcTrainSubsets(
    const TVector<NCB::TArraySubsetIndexing<ui32>>& testSubsets,
    ui32 groupCount);


template <class TDataProvidersTemplate> // TDataProvidersTemplate<...> or TTrainingDataProvidersTemplate<...>
TVector<TDataProvidersTemplate> PrepareCvFolds(
    typename TDataProvidersTemplate::TDataPtr srcData,
    const TCrossValidationParams& cvParams,
    TMaybe<ui32> foldIdx, // if Nothing() - return data for all folds, if defined - return only one fold
    bool oldCvStyleSplit,
    NPar::TLocalExecutor* localExecutor) {

    // group subsets, groups maybe trivial
    TVector<NCB::TArraySubsetIndexing<ui32>> testSubsets;

    // both NCB::Split and NCB::StratifiedSplit keep objects order
    NCB::EObjectsOrder objectsOrder = NCB::EObjectsOrder::Ordered;

    if (cvParams.Stratified) {
        testSubsets = NCB::StratifiedSplit(
            *srcData->ObjectsGrouping,
            GetTargetForStratifiedSplit(*srcData),
            cvParams.FoldCount);
    } else {
        testSubsets = NCB::Split(*srcData->ObjectsGrouping, cvParams.FoldCount, oldCvStyleSplit);
    }

    // group subsets, maybe trivial
    TVector<NCB::TArraySubsetIndexing<ui32>> trainSubsets
        = CalcTrainSubsets(testSubsets, srcData->ObjectsGrouping->GetGroupCount());

    if (cvParams.Inverted) {
        testSubsets.swap(trainSubsets);
    }

    TVector<ui32> resultFolds;

    if (foldIdx) {
        resultFolds.assign(1, *foldIdx);
    } else {
        resultFolds.resize(cvParams.FoldCount);
        std::iota(resultFolds.begin(), resultFolds.end(), 0);
    }

    TVector<TDataProvidersTemplate> result(resultFolds.size());

    TVector<std::function<void()>> tasks;

    for (ui32 resultIdx : xrange(resultFolds.size())) {
        tasks.emplace_back(
            [&, resultIdx]() {
                result[resultIdx].Learn = srcData->GetSubset(
                    GetSubset(
                        srcData->ObjectsGrouping,
                        std::move(trainSubsets[resultFolds[resultIdx]]),
                        objectsOrder
                    ),
                    localExecutor
                );
            }
        );
        tasks.emplace_back(
            [&, resultIdx]() {
                result[resultIdx].Test.emplace_back(
                    srcData->GetSubset(
                        GetSubset(
                            srcData->ObjectsGrouping,
                            std::move(testSubsets[resultFolds[resultIdx]]),
                            objectsOrder
                        ),
                        localExecutor
                    )
                );
            }
        );
    }

    NCB::ExecuteTasksInParallel(&tasks, localExecutor);

    return result;

}


void CrossValidate(
    const NJson::TJsonValue& plainJsonParams,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    NCB::TDataProviderPtr data,
    const TCrossValidationParams& cvParams,
    TVector<TCVResult>* results);
