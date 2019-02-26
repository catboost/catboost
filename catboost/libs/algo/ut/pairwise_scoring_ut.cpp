#include <library/unittest/registar.h>
#include <catboost/libs/algo/pairwise_scoring.h>
#include <catboost/libs/algo/pairwise_leaves_calculation.h>
#include <catboost/libs/helpers/query_info_helper.h>

static double CalculateScore(const TVector<double>& avrg, const TVector<double>& sumDer, const TArray2D<double>& sumWeights) {
    double score = 0;
    for (int x = 0; x < sumDer.ysize(); ++x) {
        score += avrg[x] * sumDer[x];
        for (ui32 y = 0; y < sumDer.size(); ++y) {
            score -= 0.5 * avrg[x] * avrg[y] * sumWeights[x][y];
        }
    }
    return score;
}

static TPairwiseStats CalcPairwiseStats(
    const TVector<TIndexType>& singleIdx,
    TConstArrayRef<double> weightedDerivativesData,
    const TVector<TQueryInfo>& queriesInfo,
    int leafCount,
    int bucketCount)
{
    const int docCount = singleIdx.ysize();
    TVector<TIndexType> leafIndices(docCount);
    TVector<ui8> bucketIndices(docCount);
    for(int docId = 0; docId < docCount; ++docId) {
        leafIndices[docId] = singleIdx[docId] / bucketCount;
        bucketIndices[docId] = singleIdx[docId] % bucketCount;
    }

    TPairwiseStats pairwiseStats;
    pairwiseStats.DerSums = ComputeDerSums(
        weightedDerivativesData,
        leafCount,
        bucketCount,
        leafIndices,
        [&](ui32 docId) { return bucketIndices[docId]; },
        NCB::TIndexRange<int>(docCount));
    const auto flatPairs = UnpackPairsFromQueries(queriesInfo);
    const int pairCount = flatPairs.ysize();
    pairwiseStats.PairWeightStatistics = ComputePairWeightStatistics(
        flatPairs,
        leafCount,
        bucketCount,
        leafIndices,
        [&](ui32 docId) { return bucketIndices[docId]; },
        NCB::TIndexRange<int>(pairCount));
    pairwiseStats.SplitEnsembleSpec = TSplitEnsembleSpec::OneSplit(ESplitType::FloatFeature);

    return pairwiseStats;
}

static void CalculatePairwiseScoreSimple(
    const TVector<TIndexType>& singleIdx,
    TConstArrayRef<double> weightedDerivativesData,
    const TVector<TQueryInfo>& queriesInfo,
    int leafCount,
    int bucketCount,
    ESplitType splitType,
    float l2DiagReg,
    float pairwiseNonDiagReg,
    TVector<TScoreBin>* scoreBins
) {
    Y_UNUSED(splitType);
    const int docCount = singleIdx.ysize();
    TVector<int> leafIndices(docCount), bucketIndices(docCount);
    for(int docId = 0; docId < docCount; ++docId) {
        leafIndices[docId] = singleIdx[docId] / bucketCount;
        bucketIndices[docId] = singleIdx[docId] % bucketCount;
    }

    for (int splitId = 0; splitId < bucketCount - 1; ++splitId) {
        TArray2D<double> crossMatrix(2 * leafCount, 2 * leafCount);
        crossMatrix.FillZero();
        TVector<double> derSums(2 * leafCount);

        const int docCount = leafIndices.ysize();
        for (int docId = 0; docId < docCount; ++docId) {
            if (bucketIndices[docId] > splitId) {
                derSums[2 * leafIndices[docId] + 1] += weightedDerivativesData[docId];
            } else {
                derSums[2 * leafIndices[docId]] += weightedDerivativesData[docId];
            }
        }

        for (int queryId = 0; queryId < queriesInfo.ysize(); ++queryId) {
            const TQueryInfo& queryInfo = queriesInfo[queryId];
            const int begin = queryInfo.Begin;
            const int end = queryInfo.End;
            for (int docId = begin; docId < end; ++docId) {
                for (const auto& pair : queryInfo.Competitors[docId - begin]) {
                    const int winnerBucketId = bucketIndices[docId];
                    const int loserBucketId = bucketIndices[begin + pair.Id];
                    const int winnerLeafId = leafIndices[docId];
                    const int loserLeafId = leafIndices[begin + pair.Id];
                    if (winnerBucketId == loserBucketId && winnerLeafId == loserLeafId) {
                        continue;
                    }
                    int w = 2 * winnerLeafId, l = 2 * loserLeafId;
                    if (winnerBucketId > splitId) {
                        ++w;
                    }
                    if (loserBucketId > splitId) {
                        ++l;
                    }
                    crossMatrix[w][l] -= pair.Weight;
                    crossMatrix[l][w] -= pair.Weight;
                    crossMatrix[l][l] += pair.Weight;
                    crossMatrix[w][w] += pair.Weight;
                }
            }
        }
        const TVector<double> leafValues = CalculatePairwiseLeafValues(crossMatrix, derSums, l2DiagReg, pairwiseNonDiagReg);
        (*scoreBins)[splitId].D2 = 1.0;
        (*scoreBins)[splitId].DP = CalculateScore(leafValues, derSums, crossMatrix);
    }
}
Y_UNIT_TEST_SUITE(PairwiseScoringTest) {
    Y_UNIT_TEST(PairwiseScoringTestSmall) {
        TVector<TIndexType> singleIdx = {1, 2, 0, 1, 3, 2, 1, 0, 2, 3};
        const TVector<double> ders = {0.5, -0.5, 1.2, -3.2, 0.1, 0.3, -0.6, 2.5, -1.9, 0.5};
        TVector<TQueryInfo> queriesInfo = {{0, (ui32)singleIdx.size()}};
        TVector<TVector<TCompetitor>>& comps = queriesInfo[0].Competitors;
        comps.resize(ders.size());
        comps[0].push_back({1, 1});
        comps[0].push_back({3, 1});
        comps[0].push_back({4, 1});
        comps[0].push_back({7, 1});
        comps[2].push_back({6, 1});
        comps[2].push_back({9, 1});
        comps[4].push_back({2, 1});
        comps[4].push_back({5, 1});
        comps[4].push_back({8, 1});
        const int leafCount = 1;
        const int bucketCount = 4;
        const ESplitType splitType = ESplitType::FloatFeature;
        const float l2DiagReg = 0.3;
        const float pairwiseNonDiagReg = 0.1;

        TVector<TScoreBin> scoreBins1(bucketCount - 1), scoreBins2(bucketCount - 1);
        {
            TPairwiseStats pairwiseStats = CalcPairwiseStats(singleIdx, MakeArrayRef(ders.data(), ders.size()), queriesInfo, leafCount, bucketCount);
            CalculatePairwiseScore(pairwiseStats, bucketCount, l2DiagReg, pairwiseNonDiagReg, &scoreBins1);
        }
        CalculatePairwiseScoreSimple(singleIdx, MakeArrayRef(ders.data(), ders.size()), queriesInfo, leafCount, bucketCount, splitType, l2DiagReg, pairwiseNonDiagReg, &scoreBins2);

        UNIT_ASSERT_DOUBLES_EQUAL(scoreBins1[0].DP, scoreBins2[0].DP, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(scoreBins1[1].DP, scoreBins2[1].DP, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(scoreBins1[2].DP, scoreBins2[2].DP, 1e-6);
    }

    Y_UNIT_TEST(PairwiseScoringTestBig) {
        TVector<TIndexType> singleIdx = {1, 2, 0, 1, 3, 2, 1, 0, 2, 3};
        singleIdx[0] += 4;
        singleIdx[5] += 4;
        singleIdx[6] += 4;
        singleIdx[9] += 4;
        const TVector<double> ders = {0.5, -0.5, 1.2, -3.2, 0.1, 0.3, -0.6, 2.5, -1.9, 0.5};
        TVector<TQueryInfo> queriesInfo = {{0, (ui32)singleIdx.size()}};
        TVector<TVector<TCompetitor>>& comps = queriesInfo[0].Competitors;
        comps.resize(ders.size());
        comps[0].push_back({1, 1});
        comps[0].push_back({3, 1});
        comps[0].push_back({4, 1});
        comps[0].push_back({7, 1});
        comps[2].push_back({6, 1});
        comps[2].push_back({9, 1});
        comps[4].push_back({2, 1});
        comps[4].push_back({5, 1});
        comps[4].push_back({8, 1});
        const int leafCount = 2;
        const int bucketCount = 4;
        const ESplitType splitType = ESplitType::FloatFeature;
        const float l2DiagReg = 0.3;
        const float pairwiseNonDiagReg = 0.1;

        TVector<TScoreBin> scoreBins1(bucketCount - 1), scoreBins2(bucketCount - 1);
        {
            TPairwiseStats pairwiseStats = CalcPairwiseStats(singleIdx, MakeArrayRef(ders.data(), ders.size()), queriesInfo, leafCount, bucketCount);
            CalculatePairwiseScore(pairwiseStats, bucketCount, l2DiagReg, pairwiseNonDiagReg, &scoreBins1);
        }
        CalculatePairwiseScoreSimple(singleIdx, MakeArrayRef(ders.data(), ders.size()), queriesInfo, leafCount, bucketCount, splitType, l2DiagReg, pairwiseNonDiagReg, &scoreBins2);

        UNIT_ASSERT_DOUBLES_EQUAL(scoreBins1[0].DP, scoreBins2[0].DP, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(scoreBins1[1].DP, scoreBins2[1].DP, 1e-6);
        UNIT_ASSERT_DOUBLES_EQUAL(scoreBins1[2].DP, scoreBins2[2].DP, 1e-6);
    }
}
