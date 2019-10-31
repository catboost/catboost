#include "approx_calcer_multi_helpers.h"

inline void AddDersRangeMulti(
    TConstArrayRef<TIndexType> leafIndices,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TVector<double>> approx, // [dimensionIdx][columnIdx]
    TConstArrayRef<TVector<double>> approxDeltas, // [dimensionIdx][columnIdx]
    const IDerCalcer& error,
    int rowBegin,
    int rowEnd,
    bool isUpdateWeight,
    TArrayRef<TSumMulti> leafDers // [dimensionIdx]
) {
    const int approxDimension = approx.size();
    const bool useHessian = !leafDers[0].SumDer2.Data.empty();
    THessianInfo curDer2(useHessian * approxDimension, error.GetHessianType());
    TVector<double> curDer(approxDimension);
    constexpr int UnrollMaxCount = 16;
    TVector<TVector<double>> curApprox(UnrollMaxCount, TVector<double>(approxDimension));
    for (int columnIdx = rowBegin; columnIdx < rowEnd; columnIdx += UnrollMaxCount) {
        const int unrollCount = Min(UnrollMaxCount, rowEnd - columnIdx);
        SumTransposedBlocks(columnIdx, columnIdx + unrollCount, approx, approxDeltas, MakeArrayRef(curApprox));
        for (int unrollIdx : xrange(unrollCount)) {
            error.CalcDersMulti(curApprox[unrollIdx], target[columnIdx + unrollIdx], weight.empty() ? 1 : weight[columnIdx + unrollIdx], &curDer, useHessian ? &curDer2 : nullptr);
            TSumMulti& curLeafDers = leafIndices.empty() ? leafDers[0] : leafDers[leafIndices[columnIdx + unrollIdx]];
            if (useHessian) {
                curLeafDers.AddDerDer2(curDer, curDer2);
            } else {
                curLeafDers.AddDerWeight(curDer, weight.empty() ? 1 : weight[columnIdx + unrollIdx], isUpdateWeight);
            }
        }
    }
}

void CalcLeafDersMulti(
    const TVector<TIndexType>& indices,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDeltas,
    const IDerCalcer& error,
    int sampleCount,
    bool isUpdateWeight,
    ELeavesEstimation estimationMethod,
    NPar::TLocalExecutor* localExecutor,
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
    TVector<TVector<double>>* curLeafValues
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

void CalcLeafDeltasMulti(
    const TVector<TSumMulti>& leafDer,
    ELeavesEstimation estimationMethod,
    float l2Regularizer,
    double sumAllWeights,
    int docCount,
    TVector<double>* curLeafValues
) {
    Y_ASSERT(leafDer.ysize() == 1);
    TVector<double> curDelta;
    if (estimationMethod == ELeavesEstimation::Newton) {
            CalcDeltaNewtonMulti(leafDer[0], l2Regularizer, sumAllWeights, docCount, &curDelta);
            for (int dim = 0; dim < curDelta.ysize(); ++dim) {
                (*curLeafValues)[dim] = curDelta[dim];
            }
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
            CalcDeltaGradientMulti(leafDer[0], l2Regularizer, sumAllWeights, docCount, &curDelta);
            for (int dim = 0; dim < curDelta.ysize(); ++dim) {
                (*curLeafValues)[dim]= curDelta[dim];
            }
    }
}

void UpdateApproxDeltasMulti(
    const TVector<TIndexType>& indices,
    int docCount,
    TConstArrayRef<TVector<double>> leafDeltas, //leafDeltas[dimension][leafId]
    TVector<TVector<double>>* approxDeltas,
    NPar::TLocalExecutor* localExecutor
) {
    const auto indicesRef = MakeArrayRef(indices);
    for (int dim = 0; dim < leafDeltas.ysize(); ++dim) {
        auto approxDeltaRef = MakeArrayRef((*approxDeltas)[dim]);
        auto leafDeltaRef = MakeConstArrayRef((leafDeltas)[dim]);
        NPar::ParallelFor(
            *localExecutor,
            /*from*/0,
            /*to*/ docCount,
            /*body*/[=] (int z) {
                approxDeltaRef[z] += leafDeltaRef[indicesRef[z]];
            }
        );
    }
}

void UpdateApproxDeltasMulti(
    const TVector<TIndexType>& /*indices*/,
    int docCount,
    TConstArrayRef<double> leafDeltas, //leafDeltas[dimension]
    TVector<TVector<double>>* approxDeltas,
    NPar::TLocalExecutor* localExecutor
) {
    for (int dim = 0; dim < leafDeltas.ysize(); ++dim) {
        auto approxDeltaRef = MakeArrayRef((*approxDeltas)[dim]);
        NPar::ParallelFor(
            *localExecutor,
            /*from*/ 0,
            /*to*/ docCount,
            /*body*/ [=](int z) {
                approxDeltaRef[z] += leafDeltas[dim];
            }
        );
    }
}
