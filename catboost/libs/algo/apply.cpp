#include "apply.h"
#include "greedy_tensor_search.h"
#include "target_classifier.h"
#include "full_features.h"
#include "eval_helpers.h"
#include "learn_context.h"

void CalcApproxForTree(const TAllFeatures& features,
                       const TFullModel& model,
                       int treeIdx,
                       const TCommonContext& ctx,
                       yvector<yvector<double>>* resultPtr) {
    yvector<yvector<double>>& approx = *resultPtr;

    const int approxDimension = approx.ysize();
    yvector<TIndexType> indices = BuildIndices(model.TreeStruct[treeIdx],
                                               model,
                                               features,
                                               ctx);
    const int docCount = indices.ysize();
    for (int dim = 0; dim < approxDimension; ++dim) {
        for (int doc = 0; doc < docCount; ++doc) {
            approx[dim][doc] += model.LeafValues[treeIdx][dim][indices[doc]];
        }
    }
}

yvector<yvector<double>> MapFunctionToTrees(const TFullModel& model,
                                            const TAllFeatures& features,
                                            int begin,
                                            int end,
                                            const TTreeFunction& treeFunction,
                                            int resultDimension,
                                            TCommonContext* ctx) {
    if (begin == 0 && end == 0) {
        end = model.TreeStruct.ysize();
    } else {
        end = Min(end, model.TreeStruct.ysize());
    }
    const int docCount = GetDocCount(features);

    yvector<yvector<yvector<double>>> result(CB_THREAD_LIMIT, yvector<yvector<double>>(resultDimension, yvector<double>(docCount)));

    for (int treeBlockIdx = begin; treeBlockIdx < end; treeBlockIdx += CB_THREAD_LIMIT) {
        const int nextBlockIdx = Min(end, treeBlockIdx + CB_THREAD_LIMIT);
        ctx->LocalExecutor.ExecRange([&](int treeIdx) {
            treeFunction(features, model, treeIdx, *ctx, &result[treeIdx - treeBlockIdx]);
        },
                                     treeBlockIdx, nextBlockIdx, NPar::TLocalExecutor::WAIT_COMPLETE);
    }
    for (int threadIdx = 1; threadIdx < CB_THREAD_LIMIT; ++threadIdx) {
        for (int dim = 0; dim < resultDimension; ++dim) {
            for (int doc = 0; doc < docCount; ++doc) {
                result[0][dim][doc] += result[threadIdx][dim][doc];
            }
        }
    }

    return result[0];
}

yvector<yvector<double>> ApplyModelMulti(const TFullModel& model,
                                         const TPool& pool,
                                         bool verbose,
                                         const EPredictionType predictionType,
                                         int begin, /*=0*/
                                         int end,   /*= 0*/
                                         int threadCount /*= 1*/) {
    CB_ENSURE(!pool.Docs.empty(), "Pool should not be empty");
    CB_ENSURE(model.CtrCalcerData.LearnCtrs.empty() || !pool.CatFeatures.empty(), "if model has cat-features pool should also have them");
    if (verbose) {
        SetVerboseLogingMode();
    } else {
        SetSilentLogingMode();
    }

    const int featureCount = pool.Docs[0].Factors.ysize();
    CB_ENSURE(featureCount == model.FeatureCount, "train and test datasets should have the same feature count");
    NJson::TJsonValue jsonParams = ReadTJsonValue(model.ModelInfo.at("params"));
    jsonParams.InsertValue("thread_count", threadCount);
    TCommonContext ctx(jsonParams, Nothing(), Nothing(), featureCount, pool.CatFeatures, pool.FeatureId);

    CB_ENSURE(IsClassificationLoss(ctx.Params.LossFunction) || predictionType == EPredictionType::RawFormulaVal,
              "This prediction type is supported only for classification: " << ToString<EPredictionType>(predictionType));

    TAllFeatures features;
    PrepareAllFeatures(pool.Docs, ctx.CatFeatures, model.Borders, yvector<int>(), LearnNotSet, ctx.Params.OneHotMaxSize, ctx.Params.NanMode, ctx.LocalExecutor, &features);

    int approxDimension = model.ApproxDimension;
    yvector<yvector<double>> approx = MapFunctionToTrees(model, features, begin, end, CalcApproxForTree, approxDimension, &ctx);
    approx = PrepareEval(predictionType, approx, &ctx.LocalExecutor);
    SetSilentLogingMode();
    return approx;
}

yvector<double> ApplyModel(const TFullModel& model,
                           const TPool& pool,
                           bool verbose,
                           const EPredictionType predictionType,
                           int begin, /*=0*/
                           int end,   /*= 0*/
                           int threadCount /*= 1*/) {
    return ApplyModelMulti(model, pool, verbose, predictionType, begin, end, threadCount)[0];
}
