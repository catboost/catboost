#include "apply.h"
#include "target_classifier.h"
#include "learn_context.h"

#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/model/model_pool_compatibility.h>


TVector<TVector<double>> ApplyModelMulti(const TFullModel& model,
                                         const TPool& pool,
                                         const EPredictionType predictionType,
                                         int begin, /*= 0*/
                                         int end,   /*= 0*/
                                         NPar::TLocalExecutor& executor) {
    CB_ENSURE(!pool.IsQuantized(), "Not supported for quantized pools");
    THashMap<int, int> columnReorderMap;
    CheckModelAndPoolCompatibility(model, pool, &columnReorderMap);
    const int docCount = (int)pool.Docs.GetDocCount();
    auto approxDimension = model.ObliviousTrees.ApproxDimension;
    TVector<double> approxFlat(static_cast<unsigned long>(docCount * approxDimension));

    if (docCount > 0) {
        const int threadCount = executor.GetThreadCount() + 1; //one for current thread
        const int MinBlockSize = ceil(10000.0 / sqrt(end - begin + 1)); // for 1 iteration it will be 7k docs, for 10k iterations it will be 100 docs.
        const int effectiveBlockCount = Min(threadCount, (int)ceil(docCount * 1.0 / MinBlockSize));

        NPar::TLocalExecutor::TExecRangeParams blockParams(0, docCount);
        blockParams.SetBlockCount(effectiveBlockCount);

        if (end == 0) {
            end = model.GetTreeCount();
        } else {
            end = Min<int>(end, model.GetTreeCount());
        }

        executor.ExecRange([&](int blockId) {
            TVector<TConstArrayRef<float>> repackedFeatures;
            const int blockFirstId = blockParams.FirstId + blockId * blockParams.GetBlockSize();
            const int blockLastId = Min(blockParams.LastId, blockFirstId + blockParams.GetBlockSize());
            CB_ENSURE((size_t)pool.Docs.GetEffectiveFactorCount() >= model.ObliviousTrees.GetFlatFeatureVectorExpectedSize());
            if (columnReorderMap.empty()) {
                for (size_t i = 0; i < model.ObliviousTrees.GetFlatFeatureVectorExpectedSize(); ++i) {
                    repackedFeatures.emplace_back(MakeArrayRef(pool.Docs.Factors[i].data() + blockFirstId, blockLastId - blockFirstId));
                }
            } else {
                for (size_t i = 0; i < model.ObliviousTrees.GetFlatFeatureVectorExpectedSize(); ++i) {
                    repackedFeatures.emplace_back(MakeArrayRef(pool.Docs.Factors[columnReorderMap[i]].data() + blockFirstId, blockLastId - blockFirstId));
                }
            }
            TArrayRef<double> resultRef(approxFlat.data() + blockFirstId * approxDimension, (blockLastId - blockFirstId) * approxDimension);
            model.CalcFlatTransposed(repackedFeatures, begin, end, resultRef);
        }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
    }

    TVector<TVector<double>> approx(approxDimension, TVector<double>(docCount));
    if (approxDimension == 1) { //shortcut
        approx[0].swap(approxFlat);
    } else {
        for (int dim = 0; dim < approxDimension; ++dim) {
            for (int doc = 0; doc < docCount; ++doc) {
                approx[dim][doc] = approxFlat[approxDimension * doc + dim];
            };
        }
    }

    if (predictionType == EPredictionType::InternalRawFormulaVal) {
        //shortcut
        return approx;
    } else {
        return PrepareEvalForInternalApprox(predictionType, model, approx, &executor);
    }
}


TVector<TVector<double>> ApplyModelMulti(const TFullModel& model,
                                         const TPool& pool,
                                         bool verbose,
                                         const EPredictionType predictionType,
                                         int begin,
                                         int end,
                                         int threadCount) {
    if (verbose) {
        SetVerboseLogingMode();
    } else {
        SetSilentLogingMode();
    }

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    const auto& result = ApplyModelMulti(model, pool, predictionType, begin, end, executor);
    SetSilentLogingMode();
    return result;
}

TVector<double> ApplyModel(const TFullModel& model,
                           const TPool& pool,
                           bool verbose,
                           const EPredictionType predictionType,
                           int begin, /*= 0*/
                           int end,   /*= 0*/
                           int threadCount /*= 1*/) {
    return ApplyModelMulti(model, pool, verbose, predictionType, begin, end, threadCount)[0];
}


void TModelCalcerOnPool::ApplyModelMulti(const EPredictionType predictionType, int begin, int end, TVector<double>* flatApproxBuffer, TVector<TVector<double>>* approx) {

    const int docCount = Pool.Docs.GetDocCount();
    auto approxDimension = Model.ObliviousTrees.ApproxDimension;
    TVector<double>& approxFlat = *flatApproxBuffer;
    approxFlat.resize(static_cast<unsigned long>(docCount * approxDimension)); // TODO(annaveronika): yresize?

    if (end == 0) {
        end = Model.GetTreeCount();
    } else {
        end = Min<int>(end, Model.GetTreeCount());
    }

    Executor.ExecRange([&](int blockId) {
        auto& calcer = *ThreadCalcers[blockId];
        const int blockFirstId = BlockParams.FirstId + blockId * BlockParams.GetBlockSize();
        const int blockLastId = Min(BlockParams.LastId, blockFirstId + BlockParams.GetBlockSize());
        TArrayRef<double> resultRef(approxFlat.data() + blockFirstId * approxDimension, (blockLastId - blockFirstId) * approxDimension);
        calcer.Calc(begin, end, resultRef);
    }, 0, BlockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

    approx->resize(approxDimension);

    if (approxDimension == 1) { //shortcut
        (*approx)[0].swap(approxFlat);
    } else {
        for (auto& approxProjection : *approx) {
            approxProjection.clear();
            approxProjection.resize(docCount);
        }
        for (int dim = 0; dim < approxDimension; ++dim) {
            for (int doc = 0; doc < docCount; ++doc) {
                (*approx)[dim][doc] = approxFlat[approxDimension * doc + dim];
            };
        }
    }

    if (predictionType == EPredictionType::InternalRawFormulaVal) {
        //shortcut
        return;
    } else {
        (*approx) = PrepareEvalForInternalApprox(predictionType, Model, *approx, &Executor);
    }
    flatApproxBuffer->clear();
}
