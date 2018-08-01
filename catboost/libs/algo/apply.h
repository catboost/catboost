#pragma once

#include "index_calcer.h"

#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/formula_evaluator.h>
#include <catboost/libs/model/model_pool_compatibility.h>

#include <util/generic/vector.h>

TVector<TVector<double>> ApplyModelMulti(const TFullModel& model,
                                         const TPool& pool,
                                         const EPredictionType predictionType,
                                         int begin,
                                         int end,
                                         NPar::TLocalExecutor& executor);


TVector<TVector<double>> ApplyModelMulti(const TFullModel& model,
                                         const TPool& pool,
                                         bool verbose = false,
                                         const EPredictionType predictionType = EPredictionType::RawFormulaVal,
                                         int begin = 0,
                                         int end = 0,
                                         int threadCount = 1);

TVector<double> ApplyModel(const TFullModel& model,
                           const TPool& pool,
                           bool verbose = false,
                           const EPredictionType predictionType = EPredictionType::RawFormulaVal,
                           int begin = 0,
                           int end = 0,
                           int threadCount = 1);


/*
 * Tradeoff memory for speed
 * Don't use if you need to compute model only once and on all features
 */
class TModelCalcerOnPool {
public:
    TModelCalcerOnPool(const TFullModel& model,
                       const TPool& pool,
                       NPar::TLocalExecutor& executor)
            : Model(model)
            , Pool(pool)
            , Executor(executor)
            , BlockParams(0, pool.Docs.GetDocCount()) {
        CB_ENSURE(!pool.IsQuantized(), "Not supported for quantized pools");
        CheckModelAndPoolCompatibility(model, pool);

        const int threadCount = executor.GetThreadCount() + 1; //one for current thread
        BlockParams.SetBlockCount(threadCount);
        ThreadCalcers.resize(BlockParams.GetBlockCount());

        executor.ExecRange([&](int blockId) {
            TVector<TConstArrayRef<float>> repackedFeatures;
            const int blockFirstId = BlockParams.FirstId + blockId * BlockParams.GetBlockSize();
            const int blockLastId = Min(BlockParams.LastId, blockFirstId + BlockParams.GetBlockSize());
            for (int i = 0; i < pool.Docs.GetEffectiveFactorCount(); ++i) {
                repackedFeatures.emplace_back(MakeArrayRef(pool.Docs.Factors[i].data() + blockFirstId, blockLastId - blockFirstId));
            }
            auto floatAccessor = [&repackedFeatures](const TFloatFeature& floatFeature, size_t index) -> float {
                return repackedFeatures[floatFeature.FlatFeatureIndex][index];
            };

            auto catAccessor = [&repackedFeatures](const TCatFeature& catFeature, size_t index) -> int {
                return ConvertFloatCatFeatureToIntHash(repackedFeatures[catFeature.FlatFeatureIndex][index]);
            };
            ui64 docCount = repackedFeatures[0].Size();
            ThreadCalcers[blockId] = MakeHolder<TFeatureCachedTreeEvaluator>(Model, floatAccessor, catAccessor, docCount);
        }, 0, BlockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

    }

    void ApplyModelMulti(const EPredictionType predictionType,
                         int begin, /*= 0*/
                         int end,
                         TVector<double>* flatApproxBuffer,
                         TVector<TVector<double>>* approx);
private:
    const TFullModel& Model;
    const TPool& Pool;
    NPar::TLocalExecutor& Executor;
    NPar::TLocalExecutor::TExecRangeParams BlockParams;
    TVector<THolder<TFeatureCachedTreeEvaluator>> ThreadCalcers;
};
