#include "eval_additive_metric_with_leaves.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <library/fast_exp/fast_exp.h>

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
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    const IMetric& error,
    NPar::TLocalExecutor* localExecutor
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

        const bool isObjectwise = error.GetErrorType() == EErrorType::PerObjectError;
        CB_ENSURE(isObjectwise || !queriesInfo.empty(), "Need queries to evaluate metric " + error.GetDescription());
        int maxApproxBlockSize = 4096;
        if (!isObjectwise) {
            const auto maxQuerySize = MaxElementBy(
                queriesInfo.begin() + from,
                queriesInfo.begin() + to,
                [] (const auto& query) { return query.GetSize(); })->GetSize();
            maxApproxBlockSize = Max<int>(maxApproxBlockSize, maxQuerySize);
        }
        TVector<TVector<double>> approxDeltaBlock;
        ResizeRank2(approxDimension, maxApproxBlockSize, approxDeltaBlock);

        TVector<TQueryInfo> localQueriesInfo(queriesInfo.begin(), queriesInfo.end());

        TMetricHolder result;
        for (int idx = from; idx < to; /*see below*/) {
            const int nextIdx = GetNextIdx(isObjectwise, queriesInfo, to, maxApproxBlockSize, idx);
            const int approxBlockSize = GetBlockSize(isObjectwise, queriesInfo, idx, nextIdx);
            const int approxBlockStart = isObjectwise ? idx : queriesInfo[idx].Begin;

            for (auto dimensionIdx : xrange(approxDimension)) {
                const auto approxDimension = MakeArrayRef(approxDeltaBlock[dimensionIdx]);
                const auto deltaDimension = MakeArrayRef(localLeafDelta[dimensionIdx]);
                for (auto jdx : xrange(approxBlockSize)) {
                    approxDimension[jdx] = deltaDimension[indices[approxBlockStart + jdx]];
                }
            }
            for (auto dimensionIdx : xrange(approxDimension)) {
                approxBlock[dimensionIdx] = MakeArrayRef(approx[dimensionIdx].data() + approxBlockStart, approxBlockSize);
            }
            const auto targetBlock = MakeArrayRef(target.data() + approxBlockStart, approxBlockSize);
            const auto weightBlock = weight.empty() ? TArrayRef<float>{}
                : MakeArrayRef(weight.data() + approxBlockStart, approxBlockSize);

            auto queriesInfoBlock = TArrayRef<TQueryInfo>{};
            if (!isObjectwise) {
                queriesInfoBlock = MakeArrayRef(localQueriesInfo.data() + idx, nextIdx - idx);
                for (auto& query : queriesInfoBlock) {
                    query.Begin -= approxBlockStart;
                    query.End -= approxBlockStart;
                }
            }

            const auto blockResult = EvalErrors(
                approxBlock,
                To2DConstArrayRef<double>(approxDeltaBlock),
                isExpApprox,
                targetBlock,
                weightBlock,
                queriesInfoBlock,
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
        end = target.size();
        Y_VERIFY(end <= approx[0].ysize());
    } else {
        Y_VERIFY(error.GetErrorType() == EErrorType::QuerywiseError || error.GetErrorType() == EErrorType::PairwiseError);
        end = queriesInfo.size();
    }

    return ParallelEvalMetric(evalMetric, GetMinBlockSize(end - begin), begin, end, *localExecutor);
}
