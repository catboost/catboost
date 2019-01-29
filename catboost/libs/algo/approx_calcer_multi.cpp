#include "approx_calcer_multi.h"

#include "index_calcer.h"
#include "online_predictor.h"
#include "approx_updater_helpers.h"
#include "error_functions.h"

void UpdateApproxDeltasMulti(
    bool storeExpApprox,
    const TVector<TIndexType>& indices,
    int docCount,
    TVector<TVector<double>>* leafValues, //leafValues[dimension][bucketId]
    TVector<TVector<double>>* resArr
) {
    if (storeExpApprox) {
        for (int dim = 0; dim < leafValues->ysize(); ++dim) {
            ExpApproxIf(storeExpApprox, &(*leafValues)[dim]);
            for (int z = 0; z < docCount; ++z) {
                (*resArr)[dim][z] = UpdateApprox</*StoreExpApprox*/ true>((*resArr)[dim][z], (*leafValues)[dim][indices[z]]);
            }
        }
    } else {
        for (int dim = 0; dim < leafValues->ysize(); ++dim) {
            ExpApproxIf(storeExpApprox, &(*leafValues)[dim]);
            for (int z = 0; z < docCount; ++z) {
                (*resArr)[dim][z] = UpdateApprox</*StoreExpApprox*/ false>((*resArr)[dim][z], (*leafValues)[dim][indices[z]]);
            }
        }
    }
}

template <typename TCalcMethodDelta, typename TAddSampleToBucket>
void CalcApproxDeltaIterationMulti(
    TCalcMethodDelta CalcMethodDelta,
    TAddSampleToBucket AddSampleToBucket,
    const TVector<TIndexType>& indices,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TFold::TBodyTail& bt,
    const IDerCalcer& error,
    int iteration,
    float l2Regularizer,
    TVector<TSumMulti>* buckets,
    TVector<TVector<double>>* resArr,
    TVector<TVector<double>>* sumLeafValues
) {
    UpdateBucketsMulti(AddSampleToBucket, indices, target, weight, bt.Approx, *resArr, error, bt.BodyFinish, iteration, buckets);

    // compute mixed model
    const int approxDimension = resArr->ysize();
    const int leafCount = buckets->ysize();
    TVector<TVector<double>> curLeafValues(approxDimension, TVector<double>(leafCount));
    CalcMixedModelMulti(CalcMethodDelta, *buckets, l2Regularizer, bt.BodySumWeight, bt.BodyFinish, &curLeafValues);
    if (sumLeafValues != nullptr) {
        AddElementwise(curLeafValues, sumLeafValues);
    }
    UpdateApproxDeltasMulti(error.GetIsExpApprox(), indices, bt.BodyFinish, &curLeafValues, resArr);

    // compute tail
    TVector<double> curApprox(approxDimension);
    TVector<double> avrg(approxDimension);
    TVector<double> bufferDer(approxDimension);
    THessianInfo bufferDer2(approxDimension, error.GetHessianType());
    for (int z = bt.BodyFinish; z < bt.TailFinish; ++z) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            curApprox[dim] = UpdateApprox(error.GetIsExpApprox(), bt.Approx[dim][z], (*resArr)[dim][z]);
        }

        TSumMulti& bucket = (*buckets)[indices[z]];
        AddSampleToBucket(error, curApprox, target[z], weight.empty() ? 1 : weight[z], iteration,
                          &bufferDer, &bufferDer2, &bucket);

        CalcMethodDelta(bucket, l2Regularizer, bt.BodySumWeight, bt.BodyFinish, &avrg);
        ExpApproxIf(error.GetIsExpApprox(), &avrg);
        for (int dim = 0; dim < approxDimension; ++dim) {
            (*resArr)[dim][z] = UpdateApprox(error.GetIsExpApprox(), (*resArr)[dim][z], avrg[dim]);
        }
    }
}

void CalcApproxDeltaMulti(
    const TFold& ff,
    const TFold::TBodyTail& bt,
    int leafCount,
    const IDerCalcer& error,
    const TVector<TIndexType>& indices,
    TLearnContext* ctx,
    TVector<TVector<double>>* approxDelta,
    TVector<TVector<double>>* sumLeafValues
) {
    const auto& treeLearnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const int gradientIterations = treeLearnerOptions.LeavesEstimationIterations;
    const ELeavesEstimation estimationMethod = treeLearnerOptions.LeavesEstimationMethod;
    const float l2Regularizer = treeLearnerOptions.L2Reg;

    const int approxDimension = approxDelta->ysize();
    TVector<TSumMulti> buckets(leafCount, TSumMulti(approxDimension, error.GetHessianType()));
    for (int it = 0; it < gradientIterations; ++it) {
        for (auto& bucket : buckets) {
            bucket.SetZeroDers();
        }
        if (estimationMethod == ELeavesEstimation::Newton) {
            CalcApproxDeltaIterationMulti(CalcDeltaNewtonMulti, AddSampleToBucketNewtonMulti,
                                          indices, ff.LearnTarget, ff.GetLearnWeights(), bt, error, it, l2Regularizer,
                                          &buckets, approxDelta, sumLeafValues);
        } else {
            Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
            CalcApproxDeltaIterationMulti(CalcDeltaGradientMulti, AddSampleToBucketGradientMulti,
                                          indices, ff.LearnTarget, ff.GetLearnWeights(), bt, error, it, l2Regularizer,
                                          &buckets, approxDelta, sumLeafValues);
        }
    }
}

template <typename TCalcMethodDelta, typename TAddSampleToBucket>
void CalcLeafValuesIterationMulti(
    TCalcMethodDelta CalcMethodDelta,
    TAddSampleToBucket AddSampleToBucket,
    const TVector<TIndexType>& indices,
    const TVector<float>& target,
    const TVector<float>& weight,
    const IDerCalcer& error,
    int iteration,
    float l2Regularizer,
    double sumWeight,
    TVector<TSumMulti>* buckets,
    TVector<TVector<double>>* approx
) {
    int leafCount = buckets->ysize();
    int approxDimension = approx->ysize();
    int learnSampleCount = (*approx)[0].ysize();

    UpdateBucketsMulti(AddSampleToBucket, indices, target, weight, /*approx*/ TVector<TVector<double>>(), *approx, error, learnSampleCount, iteration, buckets);

    TVector<TVector<double>> curLeafValues(approxDimension, TVector<double>(leafCount));
    CalcMixedModelMulti(CalcMethodDelta, *buckets, l2Regularizer, sumWeight, learnSampleCount, &curLeafValues);

    UpdateApproxDeltasMulti(error.GetIsExpApprox(), indices, learnSampleCount, &curLeafValues, approx);
}

void CalcLeafValuesMulti(
    int leafCount,
    const IDerCalcer& error,
    const TFold& ff,
    const TVector<TIndexType>& indices,
    TLearnContext* ctx,
    TVector<TVector<double>>* leafValues
) {
    const TFold::TBodyTail& bt = ff.BodyTailArr[0];
    const int approxDimension = ff.GetApproxDimension();

    TVector<TVector<double>> approx(approxDimension);
    for (int dim = 0; dim < approxDimension; ++dim) {
        approx[dim].assign(bt.Approx[dim].begin(), bt.Approx[dim].begin() + ff.GetLearnSampleCount());
    }

    const auto& treeLearnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const int gradientIterations = treeLearnerOptions.LeavesEstimationIterations;
    TVector<TSumMulti> buckets(leafCount, TSumMulti(approxDimension, error.GetHessianType()));
    const ELeavesEstimation estimationMethod = treeLearnerOptions.LeavesEstimationMethod;
    const float l2Regularizer = treeLearnerOptions.L2Reg;
    leafValues->assign(approxDimension, TVector<double>(leafCount));
    TVector<double> avrg(approxDimension);
    for (int it = 0; it < gradientIterations; ++it) {
        for (auto& bucket : buckets) {
            bucket.SetZeroDers();
        }
        if (estimationMethod == ELeavesEstimation::Newton) {
            CalcLeafValuesIterationMulti(CalcDeltaNewtonMulti, AddSampleToBucketNewtonMulti,
                                         indices, ff.LearnTarget, ff.GetLearnWeights(), error, it, l2Regularizer,
                                         ff.GetSumWeight(), &buckets, &approx);
        } else {
            Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
            CalcLeafValuesIterationMulti(CalcDeltaGradientMulti, AddSampleToBucketGradientMulti,
                                         indices, ff.LearnTarget, ff.GetLearnWeights(), error, it, l2Regularizer,
                                         ff.GetSumWeight(), &buckets, &approx);
        }

        for (int leaf = 0; leaf < leafCount; ++leaf) {
            if (estimationMethod == ELeavesEstimation::Newton) {
                CalcDeltaNewtonMulti(buckets[leaf], l2Regularizer, bt.BodySumWeight, bt.TailFinish, &avrg);
            } else {
                CalcDeltaGradientMulti(buckets[leaf], l2Regularizer, bt.BodySumWeight, bt.TailFinish, &avrg);
            }
            for (int dim = 0; dim < approxDimension; ++dim) {
                (*leafValues)[dim][leaf] += avrg[dim];
            }
        }
    }
}
