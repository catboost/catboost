#include "pairwise_leaves_calculation.h"

#include <catboost/libs/helpers/map_merge.h>
#include <catboost/libs/helpers/matrix.h>
#include <catboost/private/libs/index_range/index_range.h>
#include <catboost/private/libs/lapack/linear_system.h>


TVector<double> CalculatePairwiseLeafValues(
    const TArray2D<double>& pairwiseWeightSums,
    const TVector<double>& derSums,
    float l2DiagReg,
    float pairwiseBucketWeightPriorReg
) {
    Y_ASSERT(pairwiseWeightSums.GetXSize() > 1);
    Y_ASSERT(pairwiseWeightSums.GetXSize() == pairwiseWeightSums.GetYSize());
    Y_ASSERT(pairwiseWeightSums.GetXSize() == derSums.size());

    const int systemSize = pairwiseWeightSums.GetXSize();
    const double cellPrior = 1.0 / systemSize;
    const double nonDiagReg = -pairwiseBucketWeightPriorReg * cellPrior;
    const double diagReg = pairwiseBucketWeightPriorReg * (1 - cellPrior) + l2DiagReg;

    TVector<double> res;
    if (systemSize == 2) {
       /* In case of 2x2 matrix we have the system of such form:
        *     / a11 -a11\ /x1\  --  / b1\
        *     \-a11  a11/ \x2/  --  \-b1/
        * It has the following solution: x1 = b1/a11, x2 = 0.
        * */
        res = {derSums[0] / (pairwiseWeightSums[0][0] + diagReg), 0.0};
        MakeZeroAverage(&res);
        return res;
    }

    TVector<double> systemMatrix((systemSize - 1) * (systemSize - 1));
    // Copy only upper triangular of the matrix as it is symmetric and another half is not referenced in potrf.
    for (int y = 0; y < systemSize - 1; ++y) {
        for (int x = 0; x < y; ++x) {
            systemMatrix[y * (systemSize - 1) + x] = pairwiseWeightSums[y][x] + nonDiagReg;
        }
        systemMatrix[y * (systemSize - 1) + y] = pairwiseWeightSums[y][y] + diagReg;
    }

    res = derSums;
    res.pop_back();
    SolveLinearSystemCholesky(&systemMatrix, &res);
    res.push_back(0.0);

    MakeZeroAverage(&res);
    return res;
}

TArray2D<double> ComputePairwiseWeightSums(
    const TVector<TQueryInfo>& queriesInfo,
    int leafCount,
    int querycount,
    const TVector<TIndexType>& indices,
    NPar::ILocalExecutor* localExecutor
) {
    const auto mapQueries = [&](const NCB::TIndexRange<int>& range, TArray2D<double>* rangeSum) {
        rangeSum->SetSizes(leafCount, leafCount);
        rangeSum->FillZero();
        for (int queryId = range.Begin; queryId < range.End; ++queryId) {
            const TQueryInfo& queryInfo = queriesInfo[queryId];
            const int begin = queryInfo.Begin;
            const int end = queryInfo.End;
            for (int docId = begin; docId < end; ++docId) {
                for (const auto& pair : queryInfo.Competitors[docId - begin]) {
                    const int winnerLeafId = indices[docId];
                    const int loserLeafId = indices[begin + pair.Id];
                    if (winnerLeafId == loserLeafId) {
                        continue;
                    }
                    (*rangeSum)[winnerLeafId][loserLeafId] -= pair.Weight;
                    (*rangeSum)[loserLeafId][winnerLeafId] -= pair.Weight;
                    (*rangeSum)[winnerLeafId][winnerLeafId] += pair.Weight;
                    (*rangeSum)[loserLeafId][loserLeafId] += pair.Weight;
                }
            }
        }
    };
    const auto mergeSums = [&](TArray2D<double>* mergedSum, const TVector<TArray2D<double>>&& rangeSums) {
        for (const auto& rangeSum : rangeSums) {
            for (int winnerIdx = 0; winnerIdx < leafCount; ++winnerIdx) {
                for (int loserIdx = 0; loserIdx < leafCount; ++loserIdx) {
                    (*mergedSum)[winnerIdx][loserIdx] += rangeSum[winnerIdx][loserIdx];
                }
            }
        }
    };
    NCB::TSimpleIndexRangesGenerator<int> rangeGenerator({0, querycount}, CeilDiv(querycount, CB_THREAD_LIMIT));
    TArray2D<double> mergedSum;
    NCB::MapMerge(localExecutor, rangeGenerator, mapQueries, mergeSums, &mergedSum);

    return mergedSum;
}
