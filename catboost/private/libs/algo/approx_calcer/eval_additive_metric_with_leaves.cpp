#include "eval_additive_metric_with_leaves.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_multi_helpers.h>

#include <library/cpp/fast_exp/fast_exp.h>

#include <util/generic/algorithm.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>


static int GetBlockSize(
    bool isObjectwise,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int idx,
    int nextIdx
) {
    if (isObjectwise) {
        return nextIdx - idx;
    }
    return queriesInfo[nextIdx - 1].End - queriesInfo[idx].Begin;
}

static int GetNextIdx(
    bool isObjectwise,
    TConstArrayRef<TQueryInfo> queriesInfo,
    int lastIdx,
    int maxApproxBlockSize,
    int idx
) {
    if (isObjectwise) {
        return Min<int>(idx + maxApproxBlockSize, lastIdx);
    }
    int objectsCount = 0;
    while (idx < lastIdx) {
        objectsCount += queriesInfo[idx].GetSize();
        if (objectsCount > maxApproxBlockSize) {
            return idx;
        }
        idx += 1;
    }
    return idx;
}

TMetricHolder EvalErrorsWithLeaves(
    const TConstArrayRef<TConstArrayRef<double>> approx,
    const TConstArrayRef<TConstArrayRef<double>> leafDelta,
    TConstArrayRef<TIndexType> indices,
    bool isExpApprox,
    const TVector<TConstArrayRef<float>>& target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    const IMetric& error,
    NPar::ILocalExecutor* localExecutor
) {
    CB_ENSURE(error.IsAdditiveMetric(), "EvalErrorsWithLeaves is not implemented for non-additive metric " + error.GetDescription());

    const auto approxDimension = approx.size();
    TVector<TVector<double>> localLeafDelta;
    ResizeRank2(approxDimension, leafDelta[0].size(), localLeafDelta);
    AssignRank2(leafDelta, &localLeafDelta);
    if (isExpApprox) {
        for (auto& deltaDimension : localLeafDelta) {
            FastExpInplace(deltaDimension.data(), deltaDimension.size());
        }
    }

    NPar::TLocalExecutor sequentialExecutor;
    const auto evalMetric = [&] (int from, int to) { // objects or queries
        TVector<TConstArrayRef<double>> approxBlock(approxDimension, TArrayRef<double>{});
        TVector<TConstArrayRef<float>> targetBlock(target.size(), TArrayRef<float>{});

        const bool isObjectwise = error.GetErrorType() == EErrorType::PerObjectError;
        CB_ENSURE(isObjectwise || !queriesInfo.empty(), "Need queries to evaluate metric " + error.GetDescription());
        constexpr size_t MaxQueryBlockSize = 4096;
        int maxApproxBlockSize = MaxQueryBlockSize;
        if (!isObjectwise) {
            const auto maxQuerySize = MaxElementBy(
                queriesInfo.begin() + from,
                queriesInfo.begin() + to,
                [] (const auto& query) { return query.GetSize(); })->GetSize();
            maxApproxBlockSize = Max<int>(maxApproxBlockSize, maxQuerySize);
        }
        TVector<TVector<double>> approxDeltaBlock;
        ResizeRank2(approxDimension, maxApproxBlockSize, approxDeltaBlock);

        TVector<TQueryInfo> queriesInfoBlock(MaxQueryBlockSize);

        TMetricHolder result;
        for (int idx = from; idx < to; /*see below*/) {
            const int nextIdx = GetNextIdx(isObjectwise, queriesInfo, to, maxApproxBlockSize, idx);
            const int approxBlockSize = GetBlockSize(isObjectwise, queriesInfo, idx, nextIdx);
            const int approxBlockStart = isObjectwise ? idx : queriesInfo[idx].Begin;

            SetApproxDeltasMulti(
                GetSlice(indices, approxBlockStart, approxBlockSize),
                approxBlockSize,
                localLeafDelta,
                &approxDeltaBlock,
                &sequentialExecutor);
            approxBlock = To2DConstArrayRef<double>(approx, approxBlockStart, approxBlockSize);
            targetBlock = To2DConstArrayRef<float>(target, approxBlockStart, approxBlockSize);
            const auto weightBlock = GetSlice(weight, approxBlockStart, approxBlockSize);

            auto queriesInfoBlockRef = GetSlice(queriesInfoBlock, 0, nextIdx - idx);
            if (!isObjectwise) {
                for (auto queryIdx : xrange(idx, nextIdx)) {
                    queriesInfoBlockRef[queryIdx - idx] = queriesInfo[queryIdx];
                    queriesInfoBlockRef[queryIdx - idx].Begin -= approxBlockStart;
                    queriesInfoBlockRef[queryIdx - idx].End -= approxBlockStart;
                }
            }

            const auto blockResult = EvalErrors(
                approxBlock,
                To2DConstArrayRef<double>(approxDeltaBlock),
                isExpApprox,
                targetBlock,
                weightBlock,
                queriesInfoBlockRef,
                error,
                &sequentialExecutor);
            result.Add(blockResult);
            idx = nextIdx;
        }
        return result;
    };

    int begin = 0;
    int end;
    if (error.GetErrorType() == EErrorType::PerObjectError) {
        end = target[0].size();
        CB_ENSURE(end <= approx[0].ysize(), "Prediction and label size do not match");
    } else {
        CB_ENSURE(
            error.GetErrorType() == EErrorType::QuerywiseError || error.GetErrorType() == EErrorType::PairwiseError,
            "Expected querywise or pairwise metric");
        end = queriesInfo.size();
    }

    CB_ENSURE(end > 0, "Not enough data to calculate metric: groupwise metric w/o group id's, or objectwise metric w/o samples");
    return ParallelEvalMetric(evalMetric, GetMinBlockSize(end - begin), begin, end, *localExecutor);
}
