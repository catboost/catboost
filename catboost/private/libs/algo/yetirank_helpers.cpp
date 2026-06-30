#include "yetirank_helpers.h"

#include "approx_updater_helpers.h"

#include <catboost/libs/metrics/dcg.h>
#include <catboost/libs/metrics/sample.h>
#include <catboost/private/libs/data_types/pair.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/loss_description.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>
#include <util/random/normal.h>
#include <util/string/cast.h>


namespace {

    enum class EYetiRankWeightsMode {
        Classic,
        DCG,
        NDCG,
        MRR,
        ERR,
        MAP
    };

    EYetiRankWeightsMode ModeFromString(const TString& mode) {
        if (mode == "Classic")
            return EYetiRankWeightsMode::Classic;
        if (mode == "NDCG")
            return EYetiRankWeightsMode::NDCG;
        if (mode == "DCG")
            return EYetiRankWeightsMode::DCG;
        if (mode == "MRR")
            return EYetiRankWeightsMode::MRR;
        if (mode == "ERR")
            return EYetiRankWeightsMode::ERR;
        if (mode == "MAP")
            return EYetiRankWeightsMode::MAP;
        CB_ENSURE(false, "Unknown weights mode " << mode);
        Y_UNREACHABLE();
    }

    enum class EYetiRankNoiseType {
        Gumbel,
        Gauss,
        No
    };

    EYetiRankNoiseType NoiseFromString(const TString& noise) {
        if (noise == "Gumbel") {
            return EYetiRankNoiseType::Gumbel;
        }
        if (noise == "Gauss") {
            return EYetiRankNoiseType::Gauss;
        }
        if (noise == "No") {
            return EYetiRankNoiseType::No;
        }
        CB_ENSURE(false, "Unknown noise type " << noise);
        Y_UNREACHABLE();
    }


    class TYetiRankPairWeightsCalcer {
    public:
        struct TConfig {
            explicit TConfig(const NCatboostOptions::TLossDescription& lossDescription)
            {
                CB_ENSURE(
                    EqualToOneOf(lossDescription.GetLossFunction(), ELossFunction::YetiRank, ELossFunction::YetiRankPairwise),
                    "Loss should be YetiRank or YetiRankPairwise"
                );
                const auto& params = lossDescription.GetLossParamsMap();
                if (auto it = params.find("mode"); it != params.end()) {
                    Mode = ModeFromString(it->second);
                }
                Decay = NCatboostOptions::GetYetiRankDecay(lossDescription);
                if (auto it = params.find("top"); it != params.end()) {
                    TopSize = FromString<int>(it->second);
                }
                if (auto it = params.find("dcg_type"); it != params.end()) {
                    DcgType = FromString<ENdcgMetricType>(it->second);
                }
                if (auto it = params.find("dcg_denominator"); it != params.end()) {
                    DcgDenominator = FromString<ENdcgDenominatorType>(it->second);
                }
                if (auto it = params.find("noise"); it != params.end()) {
                    NoiseType = NoiseFromString(it->second);
                }
                if (auto it = params.find("noise_power"); it != params.end()) {
                    NoisePower = FromString<double>(it->second);
                }
                NumNeighbors = NCatboostOptions::GetParamOrDefault(params, "num_neighbors", 1);
            }

        public:
            EYetiRankWeightsMode Mode = EYetiRankWeightsMode::Classic;
            double Decay = 0.99;
            ui32 TopSize = Max<ui32>();
            ENdcgMetricType DcgType = ENdcgMetricType::Base;
            ENdcgDenominatorType DcgDenominator = ENdcgDenominatorType::LogPosition;

            EYetiRankNoiseType NoiseType = EYetiRankNoiseType::Gumbel;
            double NoisePower = 1.0;

            int NumNeighbors = 1;
        };

    public:
        TYetiRankPairWeightsCalcer(
            const TConfig& config,
            const float* relevs,
            ui32 querySize
        )
            : Config(config)
            , Relevs(relevs)
            , QuerySize(querySize)
        {}

        void CalcWeights(const TVector<int>& permutation, TVector<TVector<float>>* competitorsWeights) {
            switch (Config.Mode) {
                case EYetiRankWeightsMode::Classic:
                    CalcWeightsClassic(permutation, competitorsWeights);
                    return;
                case EYetiRankWeightsMode::DCG:
                    CalcWeightsDCG(permutation, competitorsWeights, 1.0);
                    return;
                case EYetiRankWeightsMode::NDCG:
                    CalcWeightsDCG(permutation, competitorsWeights, 1.0 / GetIDcg());
                    return;
                case EYetiRankWeightsMode::MRR:
                    CalcWeightsMRR(permutation, competitorsWeights);
                    return;
                case EYetiRankWeightsMode::ERR:
                    CalcWeightsERR(permutation, competitorsWeights);
                    return;
                case EYetiRankWeightsMode::MAP:
                    CalcWeightsMAP(permutation, competitorsWeights);
                    return;
            }
        }

        void AddNoise(TArrayRef<double> expApproxes, TFastRng64& rand) {
            switch (Config.NoiseType) {
                case EYetiRankNoiseType::Gumbel:
                    for (ui32 docId = 0; docId < QuerySize; ++docId) {
                        const float uniformValue = rand.GenRandReal1();
                        expApproxes[docId] *= uniformValue / (1.000001f - uniformValue);
                    }
                    return;
                case EYetiRankNoiseType::Gauss:
                    for (ui32 docId = 0; docId < QuerySize; ++docId) {
                        const double gaussNoise = NormalDistribution<double>(rand, 0, Config.NoisePower);
                        expApproxes[docId] *= exp(gaussNoise);
                    }
                    return;
                case EYetiRankNoiseType::No:
                    return;
            }
        }

    private:
        const TConfig& Config;

        const float* Relevs;
        const ui32 QuerySize;

        TMaybe<double> IDcg = Nothing();

    private:
        double GetIDcg() {
            if (!IDcg.Defined()) {
                TVector<NMetrics::TSample> samples(QuerySize);
                for (ui32 i = 0; i < QuerySize; ++i) {
                    samples[i] = NMetrics::TSample(Relevs[i], 0.0);
                }
                IDcg = CalcIDcg(samples, Config.DcgType, Nothing(), Config.TopSize, Config.DcgDenominator);
            }
            return *IDcg;
        }

        void AddWeight(int firstCandidate, int secondCandidate, float pairWeight, TVector<TVector<float>>* competitorsWeights) {
            if (Relevs[firstCandidate] > Relevs[secondCandidate]) {
                (*competitorsWeights)[firstCandidate][secondCandidate] += pairWeight;
            } else if (Relevs[firstCandidate] < Relevs[secondCandidate]) {
                (*competitorsWeights)[secondCandidate][firstCandidate] += pairWeight;
            }
        }

        void CalcWeightsClassic(const TVector<int>& permutation, TVector<TVector<float>>* competitorsWeights) {
            double decayCoefficient = 1;
            for (ui32 docId = 1; docId < QuerySize; ++docId) {
                const int firstCandidate = permutation[docId - 1];
                const int secondCandidate = permutation[docId];
                const double magicConst = 0.15; // Like in GPU

                const float pairWeight = magicConst * decayCoefficient
                    * Abs(Relevs[firstCandidate] - Relevs[secondCandidate]);
                AddWeight(firstCandidate, secondCandidate, pairWeight, competitorsWeights);
                decayCoefficient *= Config.Decay;
            }
        }

        double CalcDcgValue(float relevance, ui32 position) {
            const double numerator = Config.DcgType == ENdcgMetricType::Base ? relevance : Exp2(relevance) - 1;
            const double denominator = Config.DcgDenominator == ENdcgDenominatorType::Position ? position + 1 : Log2(position + 2);
            return position < Config.TopSize ? numerator / denominator : 0.0;
        }

        void CalcWeightsDCG(const TVector<int>& permutation, TVector<TVector<float>>* competitorsWeights, double coef) {
            const ui32 topSize = Min(Config.TopSize, QuerySize);
            for (ui32 docId = 1; docId <= topSize; ++docId) {
                const ui32 bound = Config.NumNeighbors == -1 ? QuerySize : Min(QuerySize, docId + Config.NumNeighbors);
                for (ui32 neighborId = docId + 1; neighborId <= bound; ++neighborId) {
                    const int firstCandidate = permutation[docId - 1];
                    const int secondCandidate = permutation[neighborId - 1];

                    const float pairWeight = coef * (
                        + CalcDcgValue(Relevs[firstCandidate], docId - 1) + CalcDcgValue(Relevs[secondCandidate], neighborId - 1)
                        - CalcDcgValue(Relevs[firstCandidate], neighborId - 1) - CalcDcgValue(Relevs[secondCandidate], docId - 1)
                    );
                    AddWeight(firstCandidate, secondCandidate, Abs(pairWeight), competitorsWeights);
                }
            }
        }

        void CalcWeightsMRR(const TVector<int>& permutation, TVector<TVector<float>>* competitorsWeights) {
            const ui32 topSize = Min(Config.TopSize, QuerySize);
            bool wasRelevant = false;
            for (ui32 docId = 1; docId <= topSize && !wasRelevant; ++docId) {
                const int firstCandidate = permutation[docId - 1];
                const bool isFirstRelevant = Relevs[firstCandidate] > 0;
                const ui32 bound = Config.NumNeighbors == -1 ? QuerySize : Min(QuerySize, docId + Config.NumNeighbors);
                for (ui32 neighborId = docId + 1; neighborId <= bound; ++neighborId) {
                    const int secondCandidate = permutation[neighborId - 1];
                    const bool isSecondRelevant = Relevs[secondCandidate] > 0;

                    if (isFirstRelevant ^ isSecondRelevant) {
                        const float pairWeight = 1.0 / docId - 1.0 / neighborId;
                        AddWeight(firstCandidate, secondCandidate, pairWeight, competitorsWeights);
                    }
                }
                wasRelevant |= isFirstRelevant;
            }
        }

        void CalcWeightsERR(const TVector<int>& permutation, TVector<TVector<float>>* competitorsWeights) {
            const ui32 topSize = Min(Config.TopSize, QuerySize);
            double pFirstLook = 1.0;
            for (ui32 docId = 1; docId <= topSize; ++docId) {
                const int firstCandidate = permutation[docId - 1];

                double pMiddleLook = 1.0;
                double middleRR = 0.0;
                const ui32 bound = Config.NumNeighbors == -1 ? QuerySize : Min(QuerySize, docId + Config.NumNeighbors);
                for (ui32 neighborId = docId + 1; neighborId <= bound; ++neighborId) {
                    const int secondCandidate = permutation[neighborId - 1];

                    const double firstDelta = (Relevs[firstCandidate] - Relevs[secondCandidate]) / docId;
                    const double middleDelta = middleRR * (Relevs[secondCandidate] - Relevs[firstCandidate]);
                    const double secondDelta = pMiddleLook * (Relevs[secondCandidate] - Relevs[firstCandidate]) / neighborId;

                    const float pairWeight = pFirstLook * (firstDelta + middleDelta + secondDelta);
                    AddWeight(firstCandidate, secondCandidate, Abs(pairWeight), competitorsWeights);

                    middleRR += pMiddleLook * Relevs[secondCandidate] / neighborId;
                    pMiddleLook *= (1 - Relevs[secondCandidate]);
                }

                pFirstLook *= 1 - Relevs[firstCandidate];
            }
        }

        void CalcWeightsMAP(const TVector<int>& permutation, TVector<TVector<float>>* competitorsWeights) {
            const ui32 topSize = Min(Config.TopSize, QuerySize);
            for (ui32 docId = 1; docId <= topSize; ++docId) {
                const int firstCandidate = permutation[docId - 1];
                const bool isFirstRelevant = Relevs[firstCandidate] > 0;

                double sumRR = 1.0 / docId;
                const ui32 bound = Config.NumNeighbors == -1 ? QuerySize : Min(QuerySize, docId + Config.NumNeighbors);
                for (ui32 neighborId = docId + 1; neighborId <= bound; ++neighborId) {
                    const int secondCandidate = permutation[neighborId - 1];
                    const bool isSecondRelevant = Relevs[secondCandidate] > 0;

                    const float pairWeight = isFirstRelevant ^ isSecondRelevant
                        ? sumRR
                        : 0.0;
                    AddWeight(firstCandidate, secondCandidate, pairWeight, competitorsWeights);

                    if (isSecondRelevant) {
                        sumRR += 1.0 / neighborId;
                    }
                }
            }
        }

    };
}


static void GenerateYetiRankPairsForQuery(
    const double* expApproxes,
    float queryWeight,
    ui32 querySize,
    int permutationCount,
    ui64 randomSeed,
    TVector<TVector<TCompetitor>>* competitors,
    TYetiRankPairWeightsCalcer* weightsCalcer
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
        weightsCalcer->AddNoise(bootstrappedApprox, rand);

        StableSort(
            indices,
            [&](int i, int j) {
                return bootstrappedApprox[i] > bootstrappedApprox[j];
            }
        );
        weightsCalcer->CalcWeights(indices, &competitorsWeights);
    }

    // TODO(nikitxskv): Can be optimized
    for (ui32 winnerIndex = 0; winnerIndex < querySize; ++winnerIndex) {
        for (ui32 loserIndex = 0; loserIndex < querySize; ++loserIndex) {
            const float competitorsWeight
                = queryWeight * competitorsWeights[winnerIndex][loserIndex] / permutationCount;
            if (competitorsWeight != 0) {
                competitorsRef[winnerIndex].push_back({loserIndex, competitorsWeight});
            }
        }
    }
}

void UpdatePairsForYetiRank(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> relevances,
    const NCatboostOptions::TLossDescription& lossDescription,
    ui64 randomSeed,
    int queryBegin,
    int queryEnd,
    TVector<TQueryInfo>* queriesInfo,
    NPar::ILocalExecutor* localExecutor
) {
    const TYetiRankPairWeightsCalcer::TConfig config(lossDescription);

    const int permutationCount = NCatboostOptions::GetYetiRankPermutations(lossDescription);

    NPar::ILocalExecutor::TExecRangeParams blockParams(queryBegin, queryEnd);
    blockParams.SetBlockCount(CB_THREAD_LIMIT);
    const int blockSize = blockParams.GetBlockSize();
    const ui32 blockCount = blockParams.GetBlockCount();
    const TVector<ui64> randomSeeds = GenRandUI64Vector(blockCount, randomSeed);
    NPar::ParallelFor(
        *localExecutor,
        0,
        blockCount,
        [&](int blockId) {
            TFastRng64 rand(randomSeeds[blockId]);
            const int from = queryBegin + blockId * blockSize;
            const int to = Min<int>(queryBegin + (blockId + 1) * blockSize, queryEnd);
            for (int queryIndex = from; queryIndex < to; ++queryIndex) {
                TQueryInfo& queryInfoRef = (*queriesInfo)[queryIndex];
                TYetiRankPairWeightsCalcer weightsCalcer(
                    config,
                    relevances.data() + queryInfoRef.Begin,
                    queryInfoRef.End - queryInfoRef.Begin
                );
                GenerateYetiRankPairsForQuery(
                    approxes.data() + queryInfoRef.Begin,
                    queryInfoRef.Weight,
                    queryInfoRef.End - queryInfoRef.Begin,
                    permutationCount,
                    rand.GenRand(),
                    &queryInfoRef.Competitors,
                    &weightsCalcer
                );
            }
        }
    );
}

void YetiRankRecalculation(
    const TFold& ff,
    const TFold::TBodyTail& bt,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    NPar::ILocalExecutor* localExecutor,
    TVector<TQueryInfo>* recalculatedQueriesInfo,
    TVector<float>* recalculatedPairwiseWeights
) {
    Y_ASSERT(ff.LearnTarget.size() == 1);
    *recalculatedQueriesInfo = ff.LearnQueriesInfo;
    UpdatePairsForYetiRank(
        bt.Approx[0],
        ff.LearnTarget[0],
        params.LossFunctionDescription.Get(),
        randomSeed,
        0,
        bt.TailQueryFinish,
        recalculatedQueriesInfo,
        localExecutor
    );
    recalculatedPairwiseWeights->resize(bt.PairwiseWeights.ysize());
    CalcPairwiseWeights(*recalculatedQueriesInfo, bt.TailQueryFinish, recalculatedPairwiseWeights);
}
