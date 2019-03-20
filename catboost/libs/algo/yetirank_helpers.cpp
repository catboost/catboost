#include "approx_updater_helpers.h"
#include "yetirank_helpers.h"

#include <catboost/libs/data_types/pair.h>

#include <util/generic/vector.h>

static void GenerateYetiRankPairsForQuery(
    const float* relevs,
    const double* expApproxes,
    float queryWeight,
    ui32 querySize,
    int permutationCount,
    double decaySpeed,
    ui64 randomSeed,
    TVector<TVector<TCompetitor>>* competitors
) {
    TFastRng64 rand(randomSeed);
    TVector<TVector<TCompetitor>>& competitorsRef = *competitors;
    competitorsRef.clear();
    competitorsRef.resize(querySize);

    TVector<int> indices(querySize);
    TVector<TVector<float>> competitorsWeights(querySize, TVector<float>(querySize));
    for (int permutationIndex = 0; permutationIndex < permutationCount; ++permutationIndex) {
        std::iota(indices.begin(), indices.end(), 0);
        TVector<double> bootstrappedApprox(expApproxes, expApproxes + querySize);
        for (ui32 docId = 0; docId < querySize; ++docId) {
            const float uniformValue = rand.GenRandReal1();
            // TODO(nikitxskv): try to experiment with different bootstraps.
            bootstrappedApprox[docId] *= uniformValue / (1.000001f - uniformValue);
        }

        Sort(indices, [&](int i, int j) {
            return bootstrappedApprox[i] > bootstrappedApprox[j];
        });

        double decayCoefficient = 1;
        for (ui32 docId = 1; docId < querySize; ++docId) {
            const int firstCandidate = indices[docId - 1];
            const int secondCandidate = indices[docId];
            const double magicConst = 0.15; // Like in GPU

            const float pairWeight = magicConst * decayCoefficient * Abs(relevs[firstCandidate] - relevs[secondCandidate]);
            if (relevs[firstCandidate] > relevs[secondCandidate]) {
                competitorsWeights[firstCandidate][secondCandidate] += pairWeight;
            } else if (relevs[firstCandidate] < relevs[secondCandidate]) {
                competitorsWeights[secondCandidate][firstCandidate] += pairWeight;
            }
            decayCoefficient *= decaySpeed;
        }
    }

    // TODO(nikitxskv): Can be optimized
    for (ui32 winnerIndex = 0; winnerIndex < querySize; ++winnerIndex) {
        for (ui32 loserIndex = 0; loserIndex < querySize; ++loserIndex) {
            const float competitorsWeight = queryWeight * competitorsWeights[winnerIndex][loserIndex] / permutationCount;
            if (competitorsWeight != 0) {
                competitorsRef[winnerIndex].push_back({loserIndex, competitorsWeight});
            }
        }
    }
}

void UpdatePairsForYetiRank(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> relevances,
    int queryInfoSize,
    const NCatboostOptions::TLossDescription& lossDescription,
    ui64 randomSeed,
    TVector<TQueryInfo>* queriesInfo,
    NPar::TLocalExecutor* localExecutor
) {
    const int permutationCount = NCatboostOptions::GetYetiRankPermutations(lossDescription);
    const double decaySpeed = NCatboostOptions::GetYetiRankDecay(lossDescription);

    NPar::TLocalExecutor::TExecRangeParams blockParams(0, queryInfoSize);
    blockParams.SetBlockCount(CB_THREAD_LIMIT);
    const int blockSize = blockParams.GetBlockSize();
    const ui32 blockCount = blockParams.GetBlockCount();
    const TVector<ui64> randomSeeds = GenRandUI64Vector(blockCount, randomSeed);
    NPar::ParallelFor(*localExecutor, 0, blockCount, [&](int blockId) {
        TFastRng64 rand(randomSeeds[blockId]);
        const int from = blockId * blockSize;
        const int to = Min<int>((blockId + 1) * blockSize, queryInfoSize);
        for (int queryIndex = from; queryIndex < to; ++queryIndex) {
            TQueryInfo& queryInfoRef = (*queriesInfo)[queryIndex];
            GenerateYetiRankPairsForQuery(
                relevances.data() + queryInfoRef.Begin,
                approxes.data() + queryInfoRef.Begin,
                queryInfoRef.Weight,
                queryInfoRef.End - queryInfoRef.Begin,
                permutationCount,
                decaySpeed,
                rand.GenRand(),
                &queryInfoRef.Competitors
            );
        }
    });
}

static void UpdatePairsForYetiRank(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> relevances,
    int queryInfoSize,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    TVector<TQueryInfo>* queriesInfo,
    NPar::TLocalExecutor* localExecutor
) {
    UpdatePairsForYetiRank(approxes, relevances, queryInfoSize, params.LossFunctionDescription.Get(), randomSeed, queriesInfo, localExecutor);
}

void YetiRankRecalculation(
    const TFold& ff,
    const TFold::TBodyTail& bt,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    NPar::TLocalExecutor* localExecutor,
    TVector<TQueryInfo>* recalculatedQueriesInfo,
    TVector<float>* recalculatedPairwiseWeights
) {
    *recalculatedQueriesInfo = ff.LearnQueriesInfo;
    UpdatePairsForYetiRank(
        bt.Approx[0],
        ff.LearnTarget,
        bt.TailQueryFinish,
        params,
        randomSeed,
        recalculatedQueriesInfo,
        localExecutor
    );
    recalculatedPairwiseWeights->resize(bt.PairwiseWeights.ysize());
    CalcPairwiseWeights(*recalculatedQueriesInfo, bt.TailQueryFinish, recalculatedPairwiseWeights);
}
