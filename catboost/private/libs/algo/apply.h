#pragma once

#include "features_data_helpers.h"

#include <catboost/libs/data/objects.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/model/fwd.h>
#include <catboost/private/libs/options/enums.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/ptr.h>
#include <util/generic/vector.h>

namespace NCB {
    template <class TTObjectsDataProvider>
    class TDataProviderTemplate;

    using TDataProvider = TDataProviderTemplate<TObjectsDataProvider>;
}


TVector<TVector<double>> ApplyModelMulti(
    const TFullModel& model,
    const NCB::TObjectsDataProvider& objectsData,
    const EPredictionType predictionType,
    int begin,
    int end,
    NPar::TLocalExecutor* executor = nullptr);

TVector<TVector<double>> ApplyModelMulti(
    const TFullModel& model,
    const NCB::TObjectsDataProvider& objectsData,
    bool verbose = false,
    const EPredictionType predictionType = EPredictionType::RawFormulaVal,
    int begin = 0,
    int end = 0,
    int threadCount = 1);

TVector<TVector<double>> ApplyModelMulti(
    const TFullModel& model,
    const NCB::TDataProvider& data,
    bool verbose = false,
    const EPredictionType predictionType = EPredictionType::RawFormulaVal,
    int begin = 0,
    int end = 0,
    int threadCount = 1);

TMinMax<double> ApplyModelForMinMax(
    const TFullModel& model,
    const NCB::TObjectsDataProvider& objectsData,
    int treeBegin = 0,
    int treeEnd = 0,
    NPar::TLocalExecutor* executor = nullptr);

/*
 * Tradeoff memory for speed
 * Don't use if you need to compute model only once and on all features
 */
class TModelCalcerOnPool {
public:
    TModelCalcerOnPool(
        const TFullModel& model,
        NCB::TObjectsDataProviderPtr objectsData,
        NPar::TLocalExecutor* executor);

    void ApplyModelMulti(
        const EPredictionType predictionType,
        int begin, /*= 0*/
        int end,
        TVector<double>* flatApproxBuffer,
        TVector<TVector<double>>* approx);

private:
    const TFullModel* Model;
    NCB::NModelEvaluation::TConstModelEvaluatorPtr ModelEvaluator;
    NCB::TObjectsDataProviderPtr ObjectsData;
    NPar::TLocalExecutor* Executor;
    NPar::TLocalExecutor::TExecRangeParams BlockParams;
    TVector<TIntrusivePtr<NCB::NModelEvaluation::IQuantizedData>> QuantizedDataForThreads;
};


class TLeafIndexCalcerOnPool {
public:
    TLeafIndexCalcerOnPool(
        const TFullModel& model,
        NCB::TObjectsDataProviderPtr objectsData,
        int treeStart,
        int treeEnd);

    bool Next();
    bool CanGet() const;
    TVector<NCB::NModelEvaluation::TCalcerIndexType> Get() const;

private:
    void CalcNextBatch();

private:
    const TFullModel& Model;
    NCB::NModelEvaluation::TConstModelEvaluatorPtr ModelEvaluator;

    THolder<NCB::IFeaturesBlockIterator> FeaturesBlockIterator;

    TVector<NCB::NModelEvaluation::TCalcerIndexType> CurrentBatchLeafIndexes;

    const size_t DocCount;
    const size_t TreeStart;
    const size_t TreeEnd;

    size_t CurrBatchStart;
    size_t CurrBatchSize;
    size_t CurrDocIndex;
};

TVector<ui32> CalcLeafIndexesMulti(
    const TFullModel& model,
    NCB::TObjectsDataProviderPtr objectsData,
    int treeStart = 0,
    int treeEnd = 0,
    NPar::TLocalExecutor* executor = nullptr);

TVector<ui32> CalcLeafIndexesMulti(
    const TFullModel& model,
    NCB::TObjectsDataProviderPtr objectsData,
    bool verbose = false,
    int treeStart = 0,
    int treeEnd = 0,
    int threadCount = 1);
