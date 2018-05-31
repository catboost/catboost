#include "pairwise_leaves_calculation.h"

#include <catboost/libs/helpers/matrix.h>

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
    const TVector<TIndexType>& indices
) {
    TArray2D<double> pairwiseWeightSums;
    pairwiseWeightSums.SetSizes(leafCount, leafCount);
    pairwiseWeightSums.FillZero();
    for (int queryId = 0; queryId < querycount; ++queryId) {
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
                pairwiseWeightSums[winnerLeafId][loserLeafId] -= pair.Weight;
                pairwiseWeightSums[loserLeafId][winnerLeafId] -= pair.Weight;
                pairwiseWeightSums[winnerLeafId][winnerLeafId] += pair.Weight;
                pairwiseWeightSums[loserLeafId][loserLeafId] += pair.Weight;
            }
        }
    }
    return pairwiseWeightSums;
}
