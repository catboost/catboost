#include "tensor_search_helpers.h"

#include <catboost/libs/helpers/restorable_rng.h>

static void GenerateRandomWeights(int learnSampleCount,
                            float baggingTemperature,
                            NPar::TLocalExecutor* localExecutor,
                            TRestorableFastRng64* rand,
                            TFold* fold) {
    if (baggingTemperature == 0) {
        Fill(fold->SampleWeights.begin(), fold->SampleWeights.end(), 1);
        return;
    }

    const ui64 randSeed = rand->GenRand();
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, learnSampleCount);
    blockParams.SetBlockSize(1000);
    localExecutor->ExecRange([&](int blockIdx) {
        TRestorableFastRng64 rand(randSeed + blockIdx);
        rand.Advance(10); // reduce correlation between RNGs in different threads
        float* sampleWeightsData = fold->SampleWeights.data();
        NPar::TLocalExecutor::BlockedLoopBody(blockParams, [=,&rand](int i) {
            const float w = -FastLogf(rand.GenRandReal1() + 1e-100);
            sampleWeightsData[i] = powf(w, baggingTemperature);
        })(blockIdx);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

static void CalcWeightedData(int learnSampleCount,
                        EBoostingType boostingType,
                        NPar::TLocalExecutor* localExecutor,
                        TFold* fold) {
    TFold& ff = *fold;

    const int approxDimension = ff.GetApproxDimension();
    for (TFold::TBodyTail& bt : ff.BodyTailArr) {
        int begin = 0;
        if (!IsPlainMode(boostingType)) {
            begin = bt.BodyFinish;
        }
        const float* sampleWeightsData = ff.SampleWeights.data();
        if (!bt.PairwiseWeights.empty()) {
            const float* pairwiseWeightsData = bt.PairwiseWeights.data();
            float* samplePairwiseWeightsData = bt.SamplePairwiseWeights.data();
            localExecutor->ExecRange([=](int z) {
                samplePairwiseWeightsData[z] = pairwiseWeightsData[z] * sampleWeightsData[z];
            }, NPar::TLocalExecutor::TExecRangeParams(begin, bt.TailFinish).SetBlockSize(4000)
             , NPar::TLocalExecutor::WAIT_COMPLETE);
        }
        for (int dim = 0; dim < approxDimension; ++dim) {
            const double* weightedDerivativesData = bt.WeightedDerivatives[dim].data();
            double* sampleWeightedDerivativesData = bt.SampleWeightedDerivatives[dim].data();
            localExecutor->ExecRange([=](int z) {
                sampleWeightedDerivativesData[z] = weightedDerivativesData[z] * sampleWeightsData[z];
            }, NPar::TLocalExecutor::TExecRangeParams(begin, bt.TailFinish).SetBlockSize(4000)
             , NPar::TLocalExecutor::WAIT_COMPLETE);
        }
    }

    if (!ff.LearnWeights.empty()) {
        for (int i = 0; i < learnSampleCount; ++i) {
            ff.SampleWeights[i] *= ff.LearnWeights[i];
        }
    }
}

void Bootstrap(const NCatboostOptions::TCatBoostOptions& params,
               const TVector<TIndexType>& indices,
               TFold* fold,
               TCalcScoreFold* sampledDocs,
               NPar::TLocalExecutor* localExecutor,
               TRestorableFastRng64* rand) {
    const int learnSampleCount = indices.ysize();
    const EBootstrapType bootstrapType = params.ObliviousTreeOptions->BootstrapConfig->GetBootstrapType();
    switch (bootstrapType) {
        case EBootstrapType::Bernoulli:
            Fill(fold->SampleWeights.begin(), fold->SampleWeights.end(), 1);
            break;
        case EBootstrapType::Bayesian:
            GenerateRandomWeights(learnSampleCount, params.ObliviousTreeOptions->BootstrapConfig->GetBaggingTemperature(), localExecutor, rand, fold);
            break;
        case EBootstrapType::No:
            Fill(fold->SampleWeights.begin(), fold->SampleWeights.end(), 1);
            break;
        default:
            CB_ENSURE(false, "Not supported bootstrap type on CPU: " << bootstrapType);
    }
    CalcWeightedData(learnSampleCount, params.BoostingOptions->BoostingType.Get(), localExecutor, fold);
    sampledDocs->Sample(*fold, indices, rand, localExecutor);
}

void SetBestScore(ui64 randSeed, const TVector<TVector<double>>& allScores, double scoreStDev, TVector<TCandidateInfo>* subcandidates) {
    TRestorableFastRng64 rand(randSeed);
    rand.Advance(10); // reduce correlation between RNGs in different threads
    for (size_t subcandidateIdx = 0; subcandidateIdx < allScores.size(); ++subcandidateIdx) {
        double bestScoreInstance = MINIMAL_SCORE;
        auto& splitInfo = (*subcandidates)[subcandidateIdx];
        const auto& scores = allScores[subcandidateIdx];
        for (int binFeatureIdx = 0; binFeatureIdx < scores.ysize(); ++binFeatureIdx) {
            const double score = scores[binFeatureIdx];
            const double scoreInstance = TRandomScore(score, scoreStDev).GetInstance(rand);
            if (scoreInstance > bestScoreInstance) {
                bestScoreInstance = scoreInstance;
                splitInfo.BestScore = TRandomScore(score, scoreStDev);
                splitInfo.BestBinBorderId = binFeatureIdx;
            }
        }
    }
}
