#include "approx_calcer.h"
#include "approx_calcer_helpers.h"
#include "approx_calcer_multi.h"
#include "approx_calcer_querywise.h"
#include "fold.h"
#include "score_calcer.h"
#include "index_calcer.h"
#include "learn_context.h"
#include "error_functions.h"
#include "yetirank_helpers.h"
#include "pairwise_leaves_calculation.h"

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/options/enum_helpers.h>

template <bool StoreExpApprox, int VectorWidth>
inline void UpdateApproxKernel(const double* leafValues, const TIndexType* indices, double* resArr) {
    Y_ASSERT(VectorWidth == 4);
    const TIndexType idx0 = indices[0];
    const TIndexType idx1 = indices[1];
    const TIndexType idx2 = indices[2];
    const TIndexType idx3 = indices[3];
    const double resArr0 = resArr[0];
    const double resArr1 = resArr[1];
    const double resArr2 = resArr[2];
    const double resArr3 = resArr[3];
    const double value0 = leafValues[idx0];
    const double value1 = leafValues[idx1];
    const double value2 = leafValues[idx2];
    const double value3 = leafValues[idx3];
    resArr[0] = UpdateApprox<StoreExpApprox>(resArr0, value0);
    resArr[1] = UpdateApprox<StoreExpApprox>(resArr1, value1);
    resArr[2] = UpdateApprox<StoreExpApprox>(resArr2, value2);
    resArr[3] = UpdateApprox<StoreExpApprox>(resArr3, value3);
}

template <bool StoreExpApprox>
inline void UpdateApproxBlock(
    const NPar::TLocalExecutor::TExecRangeParams& params,
    const double* leafValues,
    const TIndexType* indices,
    int blockIdx,
    double* resArr
) {
    const int blockStart = blockIdx * params.GetBlockSize();
    const int nextBlockStart = Min<ui64>(blockStart + params.GetBlockSize(), params.LastId);
    constexpr int VectorWidth = 4;
    int doc;
    for (doc = blockStart; doc + VectorWidth <= nextBlockStart; doc += VectorWidth) {
        UpdateApproxKernel<StoreExpApprox, VectorWidth>(leafValues, indices + doc, resArr + doc);
    }
    for (; doc < nextBlockStart; ++doc) {
        resArr[doc] = UpdateApprox<StoreExpApprox>(resArr[doc], leafValues[indices[doc]]);
    }
}

void UpdateApproxDeltas(
    bool storeExpApprox,
    const TVector<TIndexType>& indices,
    int docCount,
    NPar::TLocalExecutor* localExecutor,
    TVector<double>* leafValues,
    TVector<double>* resArr
) {
    ExpApproxIf(storeExpApprox, leafValues);

    double* resArrData = resArr->data();
    const TIndexType* indicesData = indices.data();
    const double* leafValuesData = leafValues->data();

    NPar::TLocalExecutor::TExecRangeParams blockParams(0, docCount);
    blockParams.SetBlockSize(1000);

    if (storeExpApprox) {
        localExecutor->ExecRange([=] (int blockIdx) {
            UpdateApproxBlock</*StoreExpApprox*/ true>(blockParams, leafValuesData, indicesData, blockIdx, resArrData);
        }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
    } else {
        localExecutor->ExecRange([=] (int blockIdx) {
            UpdateApproxBlock</*StoreExpApprox*/ false>(blockParams, leafValuesData, indicesData, blockIdx, resArrData);
        }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
    }
}

static void CalcShiftedApproxDers(
    const TVector<double>& approxes,
    const TVector<double>& approxesDelta,
    const TVector<float>& targets,
    const TVector<float>& weights,
    const IDerCalcer& error,
    int sampleStart,
    int sampleFinish,
    TVector<TDers>* weightedDers,
    TLearnContext* ctx
) {
    NPar::TLocalExecutor::TExecRangeParams blockParams(sampleStart, sampleFinish);
    blockParams.SetBlockSize(APPROX_BLOCK_SIZE);
    ctx->LocalExecutor->ExecRange([&](int blockId) {
        const int blockOffset = sampleStart + blockId * blockParams.GetBlockSize(); // espetrov: OK for small datasets
        error.CalcDersRange(
            blockOffset,
            Min(blockParams.GetBlockSize(), sampleFinish - blockOffset),
            /*calcThirdDer=*/false,
            approxes.data(),
            approxesDelta.data(),
            targets.data(),
            weights.data(),
            weightedDers->data() - sampleStart
        );
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

static void CalcApproxDersRange(
    const TVector<TIndexType>& indices,
    const TVector<float>& targets,
    const TVector<float>& weights,
    const TVector<double>& approxes,
    const TVector<double>& approxesDelta,
    const IDerCalcer& error,
    int sampleCount,
    int iteration,
    ELeavesEstimation estimationMethod,
    NPar::TLocalExecutor* localExecutor,
    TVector<TSum>* buckets,
    TVector<TDers>* weightedDers
) {
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, sampleCount);
    blockParams.SetBlockCount(CB_THREAD_LIMIT);

    const int leafCount = buckets->ysize();
    TVector<TVector<TDers>> blockBucketDers(blockParams.GetBlockCount(), TVector<TDers>(leafCount, TDers{/*Der1*/0.0, /*Der2*/0.0, /*Der3*/0.0}));
    TVector<TDers>* blockBucketDersData = blockBucketDers.data();
    // TODO(espetrov): Do not calculate sumWeights for Newton.
    // TODO(espetrov): Calculate sumWeights only on first iteration for Gradient, because on next iteration it is the same.
    // Check speedup on flights dataset.
    TVector<TVector<double>> blockBucketSumWeights(blockParams.GetBlockCount(), TVector<double>(leafCount, 0));
    TVector<double>* blockBucketSumWeightsData = blockBucketSumWeights.data();
    const TIndexType* indicesData = indices.data();
    const float* targetsData = targets.data();
    const float* weightsData = weights.data();
    const double* approxesData = approxes.data();
    const double* approxesDeltaData = approxesDelta.data();
    TDers* weightedDersData = weightedDers->data();
    localExecutor->ExecRange([=, &error](int blockId) {
        constexpr int innerBlockSize = APPROX_BLOCK_SIZE;
        TDers* approxesDer = weightedDersData + innerBlockSize * blockId;

        const int blockStart = blockId * blockParams.GetBlockSize();
        const int nextBlockStart = Min(sampleCount, blockStart + blockParams.GetBlockSize());

        TDers* bucketDers = blockBucketDersData[blockId].data();
        double* bucketSumWeights = blockBucketSumWeightsData[blockId].data();

        for (int innerBlockStart = blockStart; innerBlockStart < nextBlockStart; innerBlockStart += innerBlockSize) {
            const int nextInnerBlockStart = Min(nextBlockStart, innerBlockStart + innerBlockSize);
            error.CalcDersRange(
                innerBlockStart,
                nextInnerBlockStart - innerBlockStart,
                /*calcThirdDer=*/false,
                approxesData,
                approxesDeltaData,
                targetsData,
                weightsData,
                approxesDer - innerBlockStart
            );
            if (weightsData != nullptr) {
                for (int z = innerBlockStart; z < nextInnerBlockStart; ++z) {
                    TDers& ders = bucketDers[indicesData[z]];
                    ders.Der1 += approxesDer[z - innerBlockStart].Der1;
                    ders.Der2 += approxesDer[z - innerBlockStart].Der2;
                    bucketSumWeights[indicesData[z]] += weightsData[z];
                }
            } else {
                for (int z = innerBlockStart; z < nextInnerBlockStart; ++z) {
                    TDers& ders = bucketDers[indicesData[z]];
                    ders.Der1 += approxesDer[z - innerBlockStart].Der1;
                    ders.Der2 += approxesDer[z - innerBlockStart].Der2;
                    bucketSumWeights[indicesData[z]] += 1;
                }
            }
        }
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

    if (estimationMethod == ELeavesEstimation::Newton) {
        for (int leafId = 0; leafId < leafCount; ++leafId) {
            for (int blockId = 0; blockId < blockParams.GetBlockCount(); ++blockId) {
                if (blockBucketSumWeights[blockId][leafId] > FLT_EPSILON) {
                    UpdateBucket<ELeavesEstimation::Newton>(
                        blockBucketDers[blockId][leafId],
                        blockBucketSumWeights[blockId][leafId],
                        iteration,
                        &(*buckets)[leafId]
                    );
                }
            }
        }
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        for (int leafId = 0; leafId < leafCount; ++leafId) {
            for (int blockId = 0; blockId < blockParams.GetBlockCount(); ++blockId) {
                if (blockBucketSumWeights[blockId][leafId] > FLT_EPSILON) {
                    UpdateBucket<ELeavesEstimation::Gradient>(
                        blockBucketDers[blockId][leafId],
                        blockBucketSumWeights[blockId][leafId],
                        iteration,
                        &(*buckets)[leafId]
                    );
                }
            }
        }
    }
}

void UpdateBucketsSimple(
    const TVector<TIndexType>& indices,
    const TFold& ff,
    const TFold::TBodyTail& bt,
    const TVector<double>& approxes,
    const TVector<double>& approxDeltas,
    const IDerCalcer& error,
    int sampleCount,
    int queryCount,
    int iteration,
    ELeavesEstimation estimationMethod,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    NPar::TLocalExecutor* localExecutor,
    TVector<TSum>* buckets,
    TArray2D<double>* pairwiseBuckets,
    TVector<TDers>* scratchDers
) {
    if (error.GetErrorType() == EErrorType::PerObjectError) {
        CalcApproxDersRange(
            indices,
            ff.LearnTarget,
            ff.GetLearnWeights(),
            approxes,
            approxDeltas,
            error,
            sampleCount,
            iteration,
            estimationMethod,
            localExecutor,
            buckets,
            scratchDers
        );
    } else {
        Y_ASSERT(error.GetErrorType() == EErrorType::QuerywiseError || error.GetErrorType() == EErrorType::PairwiseError);

        TVector<TQueryInfo> recalculatedQueriesInfo;
        TVector<float> recalculatedPairwiseWeights;
        const bool shouldGenerateYetiRankPairs = ShouldGenerateYetiRankPairs(params.LossFunctionDescription->GetLossFunction());
        if (shouldGenerateYetiRankPairs) {
            YetiRankRecalculation(ff, bt, params, randomSeed, localExecutor, &recalculatedQueriesInfo, &recalculatedPairwiseWeights);
        }
        const TVector<TQueryInfo>& queriesInfo = shouldGenerateYetiRankPairs ? recalculatedQueriesInfo : ff.LearnQueriesInfo;
        const TVector<float>& weights = bt.PairwiseWeights.empty() ? ff.GetLearnWeights() : shouldGenerateYetiRankPairs ? recalculatedPairwiseWeights : bt.PairwiseWeights;

        CalculateDersForQueries(
            approxes,
            approxDeltas,
            ff.LearnTarget,
            weights,
            queriesInfo,
            error,
            /*queryStartIndex=*/0,
            queryCount,
            scratchDers,
            localExecutor
        );
        UpdateBucketsForQueries(
            *scratchDers,
            indices,
            weights,
            queriesInfo,
            /*queryStartIndex=*/0,
            queryCount,
            estimationMethod,
            iteration,
            buckets,
            localExecutor
        );
        if (IsPairwiseScoring(params.LossFunctionDescription->GetLossFunction())) {
            const int leafCount = buckets->ysize();
            *pairwiseBuckets = ComputePairwiseWeightSums(queriesInfo, leafCount, queryCount, indices, localExecutor);
        }
    }
}

void CalcMixedModelSimple(
    const TVector<TSum>& buckets,
    const TArray2D<double>& pairwiseWeightSums,
    const NCatboostOptions::TCatBoostOptions& params,
    double sumAllWeights,
    int allDocCount,
    TVector<double>* leafValues
) {
    const int leafCount = buckets.ysize();
    const float l2Regularizer = params.ObliviousTreeOptions->L2Reg;
    const float pairwiseNonDiagReg = params.ObliviousTreeOptions->PairwiseNonDiagReg;
    if (IsPairwiseScoring(params.LossFunctionDescription->GetLossFunction())) {
        TVector<double> derSums(leafCount);
        for (int leaf = 0; leaf < leafCount; ++leaf) {
            derSums[leaf] = buckets[leaf].SumDer;
        }
        *leafValues = CalculatePairwiseLeafValues(pairwiseWeightSums, derSums, l2Regularizer, pairwiseNonDiagReg);
        return;
    }

    leafValues->yresize(leafCount);
    const ELeavesEstimation estimationMethod = params.ObliviousTreeOptions->LeavesEstimationMethod;
    if (estimationMethod == ELeavesEstimation::Newton) {
        for (int leaf = 0; leaf < leafCount; ++leaf) {
            (*leafValues)[leaf] = CalcModel<ELeavesEstimation::Newton>(buckets[leaf], l2Regularizer, sumAllWeights, allDocCount);
        }
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        for (int leaf = 0; leaf < leafCount; ++leaf) {
            (*leafValues)[leaf] = CalcModel<ELeavesEstimation::Gradient>(buckets[leaf], l2Regularizer, sumAllWeights, allDocCount);
        }
    }
}

static void CalcTailModelSimple(
    const TVector<TIndexType>& indices,
    const TFold& ff,
    const TFold::TBodyTail& bt,
    const IDerCalcer& error,
    int iteration,
    float l2Regularizer,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    NPar::TLocalExecutor* localExecutor,
    TLearnContext* ctx,
    TVector<TSum>* buckets,
    TVector<double>* approxDeltas,
    TVector<TDers>* weightedDers
) {
    TVector<TQueryInfo> recalculatedQueriesInfo;
    TVector<float> recalculatedPairwiseWeights;
    const bool shouldGenerateYetiRankPairs = ShouldGenerateYetiRankPairs(params.LossFunctionDescription->GetLossFunction());
    if (shouldGenerateYetiRankPairs) {
        YetiRankRecalculation(ff, bt, params, randomSeed, localExecutor, &recalculatedQueriesInfo, &recalculatedPairwiseWeights);
    }
    const TVector<TQueryInfo>& queriesInfo = shouldGenerateYetiRankPairs ? recalculatedQueriesInfo : ff.LearnQueriesInfo;
    const TVector<float>& weights = bt.PairwiseWeights.empty() ? ff.GetLearnWeights() : shouldGenerateYetiRankPairs ? recalculatedPairwiseWeights : bt.PairwiseWeights;

    if (error.GetErrorType() == EErrorType::PerObjectError) {
        CalcShiftedApproxDers(bt.Approx[0], *approxDeltas, ff.LearnTarget, weights, error, bt.BodyFinish, bt.TailFinish, weightedDers, ctx);
    } else {
        Y_ASSERT(error.GetErrorType() == EErrorType::QuerywiseError || error.GetErrorType() == EErrorType::PairwiseError);
        CalculateDersForQueries(bt.Approx[0], *approxDeltas, ff.LearnTarget, weights, queriesInfo, error, bt.BodyQueryFinish, bt.TailQueryFinish, weightedDers, localExecutor);
    }
    TSum* bucketsData = buckets->data();
    const TIndexType* indicesData = indices.data();
    const TDers* scratchDersData = weightedDers->data();
    double* approxDeltasData = approxDeltas->data();
    TVector<double> avrg;
    avrg.yresize(1);
    const auto treeLearnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const ELeavesEstimation estimationMethod = treeLearnerOptions.LeavesEstimationMethod;
    if (estimationMethod == ELeavesEstimation::Newton) {
        double sumAllWeights = bt.BodySumWeight;
        for (int z = bt.BodyFinish; z < bt.TailFinish; ++z) {
            TSum& bucket = bucketsData[indicesData[z]];
            double w = weights.empty() ? 1 : weights[z];
            UpdateBucket<ELeavesEstimation::Newton>(scratchDersData[z - bt.BodyFinish], /*ignored weight*/0, /*ignored iteration*/-1, &bucket);
            avrg[0] = CalcModel<ELeavesEstimation::Newton>(bucket, l2Regularizer, sumAllWeights, z);
            sumAllWeights += w;
            ExpApproxIf(error.GetIsExpApprox(), &avrg);
            approxDeltasData[z] = UpdateApprox(error.GetIsExpApprox(), approxDeltasData[z], avrg[0]);
        }
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        double sumAllWeights = bt.BodySumWeight;
        for (int z = bt.BodyFinish; z < bt.TailFinish; ++z) {
            TSum& bucket = bucketsData[indicesData[z]];
            double w = weights.empty() ? 1 : weights[z];
            UpdateBucket<ELeavesEstimation::Gradient>(scratchDersData[z - bt.BodyFinish], w, iteration, &bucket);
            avrg[0] = CalcModel<ELeavesEstimation::Gradient>(bucket, l2Regularizer, sumAllWeights, z);
            sumAllWeights += w;
            ExpApproxIf(error.GetIsExpApprox(), &avrg);
            approxDeltasData[z] = UpdateApprox(error.GetIsExpApprox(), approxDeltasData[z], avrg[0]);
        }
    }
}

static void CalcApproxDeltaSimple(
    const TFold& ff,
    const TFold::TBodyTail& bt,
    int leafCount,
    const IDerCalcer& error,
    const TVector<TIndexType>& indices,
    ui64 randomSeed,
    TLearnContext* ctx,
    TVector<TVector<double>>* approxDelta,
    TVector<TVector<double>>* sumLeafValues
) {
    const int scratchSize = Max(
        !ctx->Params.BoostingOptions->ApproxOnFullHistory ? 0 : bt.TailFinish - bt.BodyFinish,
        error.GetErrorType() == EErrorType::PerObjectError ? APPROX_BLOCK_SIZE * CB_THREAD_LIMIT : bt.BodyFinish
    );
    TVector<TDers> weightedDers;
    weightedDers.yresize(scratchSize); // iteration scratch space

    const auto treeLearnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const int gradientIterations = static_cast<int>(treeLearnerOptions.LeavesEstimationIterations);
    const auto estimationMethod = treeLearnerOptions.LeavesEstimationMethod;
    const float l2Regularizer = treeLearnerOptions.L2Reg;

    TVector<TSum> buckets(leafCount, TSum()); // iteration scratch space
    TArray2D<double> pairwiseBuckets; // iteration scratch space
    TVector<double> curLeafValues; // iteration scratch space
    TVector<double>& resArr = (*approxDelta)[0];
    for (int it = 0; it < gradientIterations; ++it) {
        for (auto& bucket : buckets) {
            bucket.SetZeroDers();
        }
        UpdateBucketsSimple(indices, ff, bt, bt.Approx[0], resArr, error, bt.BodyFinish, bt.BodyQueryFinish, it, estimationMethod, ctx->Params, randomSeed, ctx->LocalExecutor, &buckets, &pairwiseBuckets, &weightedDers);
        CalcMixedModelSimple(buckets, pairwiseBuckets, ctx->Params, bt.BodySumWeight, bt.BodyFinish, &curLeafValues);
        if (sumLeafValues != nullptr) {
            AddElementwise(curLeafValues, &(*sumLeafValues)[0]);
        }

        if (!ctx->Params.BoostingOptions->ApproxOnFullHistory) {
            UpdateApproxDeltas(error.GetIsExpApprox(), indices, bt.TailFinish, ctx->LocalExecutor, &curLeafValues, &resArr);
        } else {
            Y_ASSERT(!IsPairwiseScoring(ctx->Params.LossFunctionDescription->GetLossFunction()));
            UpdateApproxDeltas(error.GetIsExpApprox(), indices, bt.BodyFinish, ctx->LocalExecutor, &curLeafValues, &resArr);
            CalcTailModelSimple(indices, ff, bt, error, it, l2Regularizer, ctx->Params, randomSeed, ctx->LocalExecutor, ctx, &buckets, &resArr, &weightedDers);
        }
    }
}

static void CalcLeafValuesSimple(
    int leafCount,
    const IDerCalcer& error,
    const TFold& ff,
    const TVector<TIndexType>& indices,
    TLearnContext* ctx,
    TVector<TVector<double>>* leafValues
) {
    const int scratchSize = error.GetErrorType() == EErrorType::PerObjectError
        ? APPROX_BLOCK_SIZE * CB_THREAD_LIMIT
        : ff.GetLearnSampleCount();
    TVector<TDers> weightedDers(scratchSize);

    const int queryCount = ff.LearnQueriesInfo.ysize();
    const auto& learnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const int gradientIterations = learnerOptions.LeavesEstimationIterations;
    const auto estimationMethod = learnerOptions.LeavesEstimationMethod;
    auto& localExecutor = *ctx->LocalExecutor;
    const TFold::TBodyTail& bt = ff.BodyTailArr[0];

    TVector<double> approxes(bt.Approx[0].begin(), bt.Approx[0].begin() + ff.GetLearnSampleCount()); // iteration scratch space
    TVector<TSum> buckets(leafCount, TSum()); // iteration scratch space
    TArray2D<double> pairwiseBuckets; // iteration scratch space
    TVector<double> curLeafValues; // iteration scratch space

    leafValues->assign(1, TVector<double>(leafCount));
    for (int it = 0; it < gradientIterations; ++it) {
        for (auto& bucket : buckets) {
            bucket.SetZeroDers();
        }
        UpdateBucketsSimple(indices, ff, bt, approxes, /*approxDeltas*/ {}, error, ff.GetLearnSampleCount(), queryCount, it, estimationMethod, ctx->Params, ctx->Rand.GenRand(), &localExecutor, &buckets, &pairwiseBuckets, &weightedDers);
        CalcMixedModelSimple(buckets, pairwiseBuckets, ctx->Params, ff.GetSumWeight(), ff.GetLearnSampleCount(), &curLeafValues);
        for (int leaf = 0; leaf < leafCount; ++leaf) {
            (*leafValues)[0][leaf] += curLeafValues[leaf];
        }
        UpdateApproxDeltas(error.GetIsExpApprox(), indices, ff.GetLearnSampleCount(), &localExecutor, &curLeafValues, &approxes);
    }
}

void CalcLeafValues(
    const NCB::TTrainingForCPUDataProviders& data,
    const IDerCalcer& error,
    const TFold& fold,
    const TSplitTree& tree,
    TLearnContext* ctx,
    TVector<TVector<double>>* leafValues,
    TVector<TIndexType>* indices
) {
    *indices = BuildIndices(fold, tree, data.Learn, data.Test, ctx->LocalExecutor);
    const int approxDimension = ctx->LearnProgress.AveragingFold.GetApproxDimension();
    Y_VERIFY(fold.GetLearnSampleCount() == data.Learn->GetObjectCount());
    const int leafCount = tree.GetLeafCount();
    if (approxDimension == 1) {
        CalcLeafValuesSimple(leafCount, error, fold, *indices, ctx, leafValues);
    } else {
        CalcLeafValuesMulti(leafCount, error, fold, *indices, ctx, leafValues);
    }
}

// output is permuted (learnSampleCount samples are permuted by LearnPermutation, test is indexed directly)
void CalcApproxForLeafStruct(
    const NCB::TTrainingForCPUDataProviders& data,
    const IDerCalcer& error,
    const TFold& fold,
    const TSplitTree& tree,
    ui64 randomSeed,
    TLearnContext* ctx,
    TVector<TVector<TVector<double>>>* approxesDelta // [bodyTailId][approxDim][docIdxInPermuted]
) {
    const TVector<TIndexType> indices = BuildIndices(fold, tree, data.Learn, data.Test, ctx->LocalExecutor);
    const int approxDimension = ctx->LearnProgress.ApproxDimension;
    const int leafCount = tree.GetLeafCount();
    TVector<ui64> randomSeeds;
    if (approxDimension == 1) {
        randomSeeds = GenRandUI64Vector(fold.BodyTailArr.ysize(), randomSeed);
    }
    approxesDelta->resize(fold.BodyTailArr.ysize());
    ctx->LocalExecutor->ExecRange([&](int bodyTailId) {
        const TFold::TBodyTail& bt = fold.BodyTailArr[bodyTailId];
        TVector<TVector<double>>& approxDelta = (*approxesDelta)[bodyTailId];
        const double initValue = GetNeutralApprox(error.GetIsExpApprox());
        if (approxDelta.empty()) {
            approxDelta.assign(approxDimension, TVector<double>(bt.TailFinish, initValue));
        } else {
            for (auto& deltaDimension : approxDelta) {
                Fill(deltaDimension.begin(), deltaDimension.end(), initValue);
            }
        }
        if (approxDimension == 1) {
            CalcApproxDeltaSimple(fold, bt, leafCount, error, indices, randomSeeds[bodyTailId], ctx, &approxDelta, /*sumLeafValues*/ nullptr);
        } else {
            CalcApproxDeltaMulti(fold, bt, leafCount, error, indices, ctx, &approxDelta, /*sumLeafValues*/ nullptr);
        }
    }, 0, fold.BodyTailArr.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
}
