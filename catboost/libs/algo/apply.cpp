#include "apply.h"
#include "target_classifier.h"
#include "full_features.h"
#include "learn_context.h"

#include <catboost/libs/helpers/eval_helpers.h>

TVector<TVector<double>> ApplyModelMulti(const TFullModel& model,
                                         const TPool& pool,
                                         const EPredictionType predictionType,
                                         int begin, /*= 0*/
                                         int end,   /*= 0*/
                                         NPar::TLocalExecutor& executor) {
    CB_ENSURE(pool.Docs.GetDocCount() != 0, "Pool should not be empty");
    const size_t poolCatFeaturesCout = pool.CatFeatures.size();
    CB_ENSURE(poolCatFeaturesCout >= model.ObliviousTrees.GetNumCatFeatures(), "Insufficient categorical features count");
    CB_ENSURE((pool.Docs.Factors.size() - poolCatFeaturesCout) >= model.GetNumFloatFeatures(), "Insufficient float features count " << (pool.Docs.Factors.size() - poolCatFeaturesCout) << "<" << model.GetNumFloatFeatures());
    const int docCount = (int)pool.Docs.GetDocCount();
    auto approxDimension = model.ObliviousTrees.ApproxDimension;
    TVector<double> approxFlat(static_cast<unsigned long>(docCount * approxDimension));
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, docCount);
    const int threadCount = executor.GetThreadCount() + 1; //one for current thread
    blockParams.SetBlockCount(threadCount);

    if (end == 0) {
        end = model.GetTreeCount();
    } else {
        end = Min<int>(end, model.GetTreeCount());
    }

    executor.ExecRange([&](int blockId) {
        TVector<TConstArrayRef<float>> repackedFeatures;
        const int blockFirstId = blockParams.FirstId + blockId * blockParams.GetBlockSize();
        const int blockLastId = Min(blockParams.LastId, blockFirstId + blockParams.GetBlockSize());
        for (int i = 0; i < pool.Docs.GetFactorsCount(); ++i) {
            repackedFeatures.emplace_back(MakeArrayRef(pool.Docs.Factors[i].data() + blockFirstId, blockLastId - blockFirstId));
        }
        TArrayRef<double> resultRef(approxFlat.data() + blockFirstId * approxDimension, (blockLastId - blockFirstId) * approxDimension);
        model.CalcFlatTransposed(repackedFeatures, begin, end, resultRef);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

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

    if (predictionType == EPredictionType::RawFormulaVal) {
        //shortcut
        return approx;
    } else {
        return PrepareEval(predictionType, approx, &executor);
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
    TVector<TVector<double>> result = ApplyModelMulti(model, pool, predictionType, begin, end, executor);
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

