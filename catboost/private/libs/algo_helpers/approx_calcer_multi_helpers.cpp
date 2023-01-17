#include "approx_calcer_multi_helpers.h"

#include <catboost/libs/helpers/dispatch_generic_lambda.h>

inline void AddDersRangeMulti(
    TConstArrayRef<TIndexType> leafIndices,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TVector<double>> approx, // [dimensionIdx][docIdx]
    TConstArrayRef<TVector<double>> approxDeltas, // [dimensionIdx][docIdx]
    const IDerCalcer& error,
    int rowBegin,
    int rowEnd,
    bool isUpdateWeight,
    TArrayRef<TSumMulti> leafDers // [dimensionIdx]
) {
    const auto* multiError = dynamic_cast<const TMultiDerCalcer*>(&error);
    const bool isMultiTarget = multiError != nullptr;

    const int approxDimension = approx.size();
    const bool useHessian = !leafDers[0].SumDer2.Data.empty();
    THessianInfo curDer2(useHessian * approxDimension, error.GetHessianType());
    TVector<double> curDer(approxDimension);
    constexpr int UnrollMaxCount = 16;
    TVector<TVector<double>> curApprox(UnrollMaxCount, TVector<double>(approxDimension));
    TVector<TVector<float>> curTarget;
    if (isMultiTarget) {
        curTarget = TVector<TVector<float>>(UnrollMaxCount, TVector<float>(target.size()));
    }

    const auto addDersRangeMultiImpl = [&](auto useWeights, auto useLeafIndices, auto useHessian, auto isMultiTarget) {
        for (int columnIdx = rowBegin; columnIdx < rowEnd; columnIdx += UnrollMaxCount) {
            const int unrollCount = Min(UnrollMaxCount, rowEnd - columnIdx);
            SumTransposedBlocks(columnIdx, columnIdx + unrollCount, approx, approxDeltas, MakeArrayRef(curApprox));
            if (isMultiTarget) {
                SumTransposedBlocks(columnIdx, columnIdx + unrollCount, target, /*targetDeltas*/{}, MakeArrayRef(curTarget));
            }
            for (int unrollIdx : xrange(unrollCount)) {
                const double w = useWeights ? weight[columnIdx + unrollIdx] : 1;

                if (isMultiTarget) {
                    multiError->CalcDers(curApprox[unrollIdx], curTarget[unrollIdx], w, &curDer, useHessian ? &curDer2 : nullptr);
                } else {
                    error.CalcDersMulti(curApprox[unrollIdx], target[0][columnIdx + unrollIdx], w, &curDer, useHessian ? &curDer2 : nullptr);
                }

                TSumMulti& curLeafDers = useLeafIndices ? leafDers[leafIndices[columnIdx + unrollIdx]] : leafDers[0];
                if (useHessian) {
                    curLeafDers.AddDerDer2(curDer, curDer2);
                } else {
                    curLeafDers.AddDerWeight(curDer, w, isUpdateWeight);
                }
            }
        }
    };

    DispatchGenericLambda(addDersRangeMultiImpl, !weight.empty(), !leafIndices.empty(), useHessian, isMultiTarget);
}

void CalcLeafDersMulti(
    const TVector<TIndexType>& indices,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDeltas,
    const IDerCalcer& error,
    int sampleCount,
    bool isUpdateWeight,
    ELeavesEstimation estimationMethod,
    NPar::ILocalExecutor* localExecutor,
    TVector<TSumMulti>* leafDers
) {
    const int approxDimension = approx.ysize();
    Y_ASSERT(approxDimension > 0);
    const auto leafCount = leafDers->size();
    for (auto& curLeafDers : *leafDers) {
        curLeafDers.SetZeroDers();
    }
    const auto& zeroDers = MakeZeroDers(approxDimension, estimationMethod, error.GetHessianType());
    const auto hessianSize = zeroDers.SumDer2.Data.size();
    NCB::MapMerge(
        localExecutor,
        NCB::TSimpleIndexRangesGenerator<int>(NCB::TIndexRange<int>(sampleCount), /*blockSize*/Max<ui32>(1000, hessianSize / CB_THREAD_LIMIT)),
        /*mapFunc*/[&](NCB::TIndexRange<int> partIndexRange, TVector<TSumMulti>* leafDers) {
            Y_ASSERT(!partIndexRange.Empty());
            leafDers->resize(leafCount, zeroDers);
            AddDersRangeMulti(
                indices,
                target,
                weight,
                approx, // [dimensionIdx][rowIdx]
                approxDeltas, // [dimensionIdx][rowIdx]
                error,
                partIndexRange.Begin,
                partIndexRange.End,
                isUpdateWeight,
                *leafDers // [dimensionIdx]
            );
        },
        /*mergeFunc*/[=](TVector<TSumMulti>* leafDers, TVector<TVector<TSumMulti>>&& addVector) {
            if (estimationMethod == ELeavesEstimation::Newton) {
                for (auto leafIdx : xrange(leafCount)) {
                    for (const auto& addItem : addVector) {
                        (*leafDers)[leafIdx].AddDerDer2(addItem[leafIdx].SumDer, addItem[leafIdx].SumDer2);
                    }
                }
            } else {
                for (auto leafIdx : xrange(leafCount)) {
                    for (const auto& addItem : addVector) {
                        (*leafDers)[leafIdx].AddDerWeight(addItem[leafIdx].SumDer, addItem[leafIdx].SumWeights, isUpdateWeight);
                    }
                }
            }
        },
        leafDers
    );
}

void CalcLeafDeltasMulti(
    const TVector<TSumMulti>& leafDers,
    ELeavesEstimation estimationMethod,
    float l2Regularizer,
    double sumAllWeights,
    int docCount,
    TVector<TVector<double>>* curLeafValues // [approxDim][leafIdx]
) {
    const int leafCount = leafDers.ysize();
    TVector<double> curDelta;
    if (estimationMethod == ELeavesEstimation::Newton) {
        for (int leaf = 0; leaf < leafCount; ++leaf) {
            CalcDeltaNewtonMulti(leafDers[leaf], l2Regularizer, sumAllWeights, docCount, &curDelta);
            for (int dim = 0; dim < curDelta.ysize(); ++dim) {
                (*curLeafValues)[dim][leaf] = curDelta[dim];
            }
        }
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        for (int leaf = 0; leaf < leafCount; ++leaf) {
            CalcDeltaGradientMulti(leafDers[leaf], l2Regularizer, sumAllWeights, docCount, &curDelta);
            for (int dim = 0; dim < curDelta.ysize(); ++dim) {
                (*curLeafValues)[dim][leaf] = curDelta[dim];
            }
        }
    }
}


template <bool isReset>
static void UpdateApproxDeltasMultiImpl(
    TConstArrayRef<TIndexType> indices, // not used if leaf count == 1
    int docCount,
    TConstArrayRef<TVector<double>> leafDeltas, // [dimension][leafId]
    TVector<TVector<double>>* approxDeltas,
    NPar::ILocalExecutor* localExecutor
) {
    NPar::ILocalExecutor::TExecRangeParams blockParams(0, docCount);
    blockParams.SetBlockSize(AdjustBlockSize(docCount, /*regularBlockSize*/1000));

    const auto indicesRef = MakeArrayRef(indices);
    const auto leafCount = leafDeltas[0].size();
    for (int dim = 0; dim < leafDeltas.ysize(); ++dim) {
        auto approxDeltaRef = MakeArrayRef((*approxDeltas)[dim]);
        if (leafCount == 1) {
            const auto delta = leafDeltas[dim][0];
            localExecutor->ExecRange(
                [=] (int z) {
                    if (isReset) {
                        approxDeltaRef[z] = delta;
                    } else {
                        approxDeltaRef[z] += delta;
                    }
                },
                blockParams,
                NPar::TLocalExecutor::WAIT_COMPLETE);
        } else {
            auto leafDeltaRef = MakeConstArrayRef(leafDeltas[dim]);
            localExecutor->ExecRange(
                [=] (int z) {
                    if (isReset) {
                        approxDeltaRef[z] = leafDeltaRef[indicesRef[z]];
                    } else {
                        approxDeltaRef[z] += leafDeltaRef[indicesRef[z]];
                    }
                },
                blockParams,
                NPar::TLocalExecutor::WAIT_COMPLETE);
        }
    }
}

void UpdateApproxDeltasMulti(
    TConstArrayRef<TIndexType> indices,
    int docCount,
    TConstArrayRef<TVector<double>> leafDeltas,
    TVector<TVector<double>>* approxDeltas,
    NPar::ILocalExecutor* localExecutor
) {
    UpdateApproxDeltasMultiImpl<false>(indices, docCount, leafDeltas, approxDeltas, localExecutor);
}

void SetApproxDeltasMulti(
    TConstArrayRef<TIndexType> indices,
    int docCount,
    TConstArrayRef<TVector<double>> leafDeltas,
    TVector<TVector<double>>* approxDeltas,
    NPar::ILocalExecutor* localExecutor
) {
    UpdateApproxDeltasMultiImpl<true>(indices, docCount, leafDeltas, approxDeltas, localExecutor);
}
