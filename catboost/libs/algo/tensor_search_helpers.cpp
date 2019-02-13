#include "tensor_search_helpers.h"

#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/helpers/restorable_rng.h>

#include <util/generic/maybe.h>


using namespace NCB;


TSplit TCandidateInfo::GetBestSplit(const TQuantizedForCPUObjectsDataProvider& objectsData) const {
    if (SplitEnsemble.IsBinarySplitsPack) {
        TPackedBinaryIndex packedBinaryIndex(SplitEnsemble.BinarySplitsPack.PackIdx, BestBinId);
        auto featureInfo = objectsData.GetPackedBinaryFeatureSrcIndex(packedBinaryIndex);
        TSplitCandidate splitCandidate;
        splitCandidate.Type
            = featureInfo.first == EFeatureType::Float ?
                ESplitType::FloatFeature :
                ESplitType::OneHotFeature;
        splitCandidate.FeatureIdx = featureInfo.second;

        return TSplit(std::move(splitCandidate), (featureInfo.first == EFeatureType::Float) ? 0 : 1);
    } else {
        return TSplit(SplitEnsemble.SplitCandidate, BestBinId);
    }
}


THolder<IDerCalcer> BuildError(
    const NCatboostOptions::TCatBoostOptions& params,
    const TMaybe<TCustomObjectiveDescriptor>& descriptor
) {
    const bool isStoreExpApprox = IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction());
    switch (params.LossFunctionDescription->GetLossFunction()) {
        case ELossFunction::Logloss:
        case ELossFunction::CrossEntropy:
            return MakeHolder<TCrossEntropyError>(isStoreExpApprox);
        case ELossFunction::RMSE:
            return MakeHolder<TRMSEError>(isStoreExpApprox);
        case ELossFunction::MAE:
        case ELossFunction::Quantile: {
            const auto& lossParams = params.LossFunctionDescription->GetLossParams();
            if (lossParams.empty()) {
                return MakeHolder<TQuantileError>(isStoreExpApprox);
            } else {
                CB_ENSURE(lossParams.begin()->first == "alpha", "Invalid loss description" << ToString(params.LossFunctionDescription.Get()));
                return MakeHolder<TQuantileError>(FromString<float>(lossParams.at("alpha")), isStoreExpApprox);
            }
        }
        case ELossFunction::LogLinQuantile: {
            const auto& lossParams = params.LossFunctionDescription->GetLossParams();
            if (lossParams.empty()) {
                return MakeHolder<TLogLinQuantileError>(isStoreExpApprox);
            } else {
                CB_ENSURE(lossParams.begin()->first == "alpha", "Invalid loss description" << ToString(params.LossFunctionDescription.Get()));
                return MakeHolder<TLogLinQuantileError>(FromString<float>(lossParams.at("alpha")), isStoreExpApprox);
            }
        }
        case ELossFunction::MAPE:
            return MakeHolder<TMAPError>(isStoreExpApprox);
        case ELossFunction::Poisson:
            return MakeHolder<TPoissonError>(isStoreExpApprox);
        case ELossFunction::MultiClass:
            return MakeHolder<TMultiClassError>(isStoreExpApprox);
        case ELossFunction::MultiClassOneVsAll:
            return MakeHolder<TMultiClassOneVsAllError>(isStoreExpApprox);
        case ELossFunction::PairLogit:
            return MakeHolder<TPairLogitError>(isStoreExpApprox);
        case ELossFunction::PairLogitPairwise:
            return MakeHolder<TPairLogitError>(isStoreExpApprox);
        case ELossFunction::QueryRMSE:
            return MakeHolder<TQueryRmseError>(isStoreExpApprox);
        case ELossFunction::QuerySoftMax: {
            const auto& lossFunctionDescription = params.LossFunctionDescription;
            const auto& lossParams = lossFunctionDescription->GetLossParams();
            CB_ENSURE(
                lossParams.empty() || lossParams.begin()->first == "lambda",
                "Invalid loss description" << ToString(lossFunctionDescription.Get())
            );

            const double lambdaReg = NCatboostOptions::GetQuerySoftMaxLambdaReg(lossFunctionDescription);
            return MakeHolder<TQuerySoftMaxError>(lambdaReg, isStoreExpApprox);
        }
        case ELossFunction::YetiRank:
            return MakeHolder<TPairLogitError>(isStoreExpApprox);
        case ELossFunction::YetiRankPairwise:
            return MakeHolder<TPairLogitError>(isStoreExpApprox);
        case ELossFunction::Lq:
            return MakeHolder<TLqError>(NCatboostOptions::GetLqParam(params.LossFunctionDescription), isStoreExpApprox);
        case ELossFunction::Custom:
            return MakeHolder<TCustomError>(params, descriptor);
        case ELossFunction::UserPerObjMetric:
            return MakeHolder<TUserDefinedPerObjectError>(params.LossFunctionDescription->GetLossParams(), isStoreExpApprox);
        case ELossFunction::UserQuerywiseMetric:
            return MakeHolder<TUserDefinedQuerywiseError>(params.LossFunctionDescription->GetLossParams(), isStoreExpApprox);
        default:
            CB_ENSURE(false, "provided error function is not supported");
    }
}

static void GenerateRandomWeights(
    int learnSampleCount,
    float baggingTemperature,
    NPar::TLocalExecutor* localExecutor,
    TRestorableFastRng64* rand,
    TFold* fold
) {
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

static void GenerateBayesianWeightsForPairs(
    float baggingTemperature,
    NPar::TLocalExecutor* localExecutor,
    TRestorableFastRng64* rand,
    TFold* fold
) {
    if (baggingTemperature == 0.0f) {
        return;
    }
    const ui64 randSeed = rand->GenRand();
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, fold->LearnQueriesInfo.ysize());
    blockParams.SetBlockSize(1000);
    localExecutor->ExecRange([&](int blockIdx) {
        TRestorableFastRng64 rand(randSeed + blockIdx);
        rand.Advance(10); // reduce correlation between RNGs in different threads
        NPar::TLocalExecutor::BlockedLoopBody(blockParams, [&](int i) {
            for (auto& competitors : fold->LearnQueriesInfo[i].Competitors) {
                for (auto& competitor : competitors) {
                    const float w = -FastLogf(rand.GenRandReal1() + 1e-100);
                    competitor.SampleWeight = competitor.Weight * powf(w, baggingTemperature);
                }
            }
        })(blockIdx);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

static void GenerateBernoulliWeightsForPairs(
    float takenFraction,
    NPar::TLocalExecutor* localExecutor,
    TRestorableFastRng64* rand,
    TFold* fold
) {
    if (takenFraction == 1.0f) {
        return;
    }
    const ui64 randSeed = rand->GenRand();
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, fold->LearnQueriesInfo.ysize());
    blockParams.SetBlockSize(1000);
    localExecutor->ExecRange([&](int blockIdx) {
        TRestorableFastRng64 rand(randSeed + blockIdx);
        rand.Advance(10); // reduce correlation between RNGs in different threads
        NPar::TLocalExecutor::BlockedLoopBody(blockParams, [&](int i) {
            for (auto& competitors : fold->LearnQueriesInfo[i].Competitors) {
                for (auto& competitor : competitors) {
                    if (rand.GenRandReal1() < takenFraction) {
                        competitor.SampleWeight = competitor.Weight;
                    } else {
                        competitor.SampleWeight = 0.0f;
                    }
                }
            }
        })(blockIdx);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

static void CalcWeightedData(
    int learnSampleCount,
    EBoostingType boostingType,
    NPar::TLocalExecutor* localExecutor,
    TFold* fold
) {
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

    const auto& learnWeights = ff.GetLearnWeights();
    if (!learnWeights.empty()) {
        for (int i = 0; i < learnSampleCount; ++i) {
            ff.SampleWeights[i] *= learnWeights[i];
        }
    }
}

void Bootstrap(
    const NCatboostOptions::TCatBoostOptions& params,
    const TVector<TIndexType>& indices,
    TFold* fold,
    TCalcScoreFold* sampledDocs,
    NPar::TLocalExecutor* localExecutor,
    TRestorableFastRng64* rand
) {
    const int learnSampleCount = indices.ysize();
    const EBootstrapType bootstrapType = params.ObliviousTreeOptions->BootstrapConfig->GetBootstrapType();
    const float baggingTemperature = params.ObliviousTreeOptions->BootstrapConfig->GetBaggingTemperature();
    const float takenFraction = params.ObliviousTreeOptions->BootstrapConfig->GetTakenFraction();
    const bool isPairwiseScoring = IsPairwiseScoring(params.LossFunctionDescription->GetLossFunction());
    switch (bootstrapType) {
        case EBootstrapType::Bernoulli:
            if (isPairwiseScoring) {
                // TODO(nikitxskv): Need to add groupwise sampling (take the whole group or not)
                GenerateBernoulliWeightsForPairs(takenFraction, localExecutor, rand, fold);
            } else {
                Fill(fold->SampleWeights.begin(), fold->SampleWeights.end(), 1);
            }
            break;
        case EBootstrapType::Bayesian:
            if (isPairwiseScoring) {
                GenerateBayesianWeightsForPairs(baggingTemperature, localExecutor, rand, fold);
            } else {
                GenerateRandomWeights(learnSampleCount, baggingTemperature, localExecutor, rand, fold);
            }
            break;
        case EBootstrapType::No:
            if (!isPairwiseScoring) {
                Fill(fold->SampleWeights.begin(), fold->SampleWeights.end(), 1);
            }
            break;
        default:
            CB_ENSURE(false, "Not supported bootstrap type on CPU: " << bootstrapType);
    }
    if (!isPairwiseScoring) {
        CalcWeightedData(learnSampleCount, params.BoostingOptions->BoostingType.Get(), localExecutor, fold);
    }
    sampledDocs->Sample(*fold, indices, rand, localExecutor);
}

void CalcWeightedDerivatives(
    const IDerCalcer& error,
    int bodyTailIdx,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    TFold* takenFold,
    NPar::TLocalExecutor* localExecutor
) {
    TFold::TBodyTail& bt = takenFold->BodyTailArr[bodyTailIdx];
    const TVector<TVector<double>>& approx = bt.Approx;
    const TVector<float>& target = takenFold->LearnTarget;
    const TVector<float>& weight = takenFold->GetLearnWeights();
    TVector<TVector<double>>* weightedDerivatives = &bt.WeightedDerivatives;

    if (error.GetErrorType() == EErrorType::QuerywiseError || error.GetErrorType() == EErrorType::PairwiseError) {
        TVector<TQueryInfo> recalculatedQueriesInfo;
        const bool shouldGenerateYetiRankPairs = ShouldGenerateYetiRankPairs(params.LossFunctionDescription->GetLossFunction());
        if (shouldGenerateYetiRankPairs) {
            YetiRankRecalculation(*takenFold, bt, params, randomSeed, localExecutor, &recalculatedQueriesInfo, &bt.PairwiseWeights);
        }
        const TVector<TQueryInfo>& queriesInfo = shouldGenerateYetiRankPairs ? recalculatedQueriesInfo : takenFold->LearnQueriesInfo;

        const int tailQueryFinish = bt.TailQueryFinish;
        TVector<TDers> ders((*weightedDerivatives)[0].ysize());
        error.CalcDersForQueries(0, tailQueryFinish, approx[0], target, weight, queriesInfo, ders, localExecutor);
        for (int docId = 0; docId < ders.ysize(); ++docId) {
            (*weightedDerivatives)[0][docId] = ders[docId].Der1;
        }
        if (params.LossFunctionDescription->GetLossFunction() == ELossFunction::YetiRankPairwise) {
            // In case of YetiRankPairwise loss function we need to store generated pairs for tree structure building.
            Y_ASSERT(takenFold->BodyTailArr.size() == 1);
            takenFold->LearnQueriesInfo.swap(recalculatedQueriesInfo);
        }
    } else {
        const int tailFinish = bt.TailFinish;
        const int approxDimension = approx.ysize();
        NPar::TLocalExecutor::TExecRangeParams blockParams(0, tailFinish);
        blockParams.SetBlockSize(1000);

        Y_ASSERT(error.GetErrorType() == EErrorType::PerObjectError);
        if (approxDimension == 1) {
            localExecutor->ExecRange([&](int blockId) {
                const int blockOffset = blockId * blockParams.GetBlockSize();
                error.CalcFirstDerRange(blockOffset, Min<int>(blockParams.GetBlockSize(), tailFinish - blockOffset),
                    approx[0].data(),
                    nullptr, // no approx deltas
                    target.data(),
                    weight.data(),
                    (*weightedDerivatives)[0].data());
            }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
        } else {
            localExecutor->ExecRange([&](int blockId) {
                TVector<double> curApprox(approxDimension);
                TVector<double> curDelta(approxDimension);
                NPar::TLocalExecutor::BlockedLoopBody(blockParams, [&](int z) {
                    for (int dim = 0; dim < approxDimension; ++dim) {
                        curApprox[dim] = approx[dim][z];
                    }
                    error.CalcDersMulti(curApprox, target[z], weight.empty() ? 1 : weight[z], &curDelta, nullptr);
                    for (int dim = 0; dim < approxDimension; ++dim) {
                        (*weightedDerivatives)[dim][z] = curDelta[dim];
                    }
                })(blockId);
            }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
        }
    }
}

void SetBestScore(
    ui64 randSeed,
    const TVector<TVector<double>>& allScores,
    double scoreStDev,
    TConstArrayRef<NCB::TBinaryFeaturesPack> perPackMasks,
    TVector<TCandidateInfo>* subcandidates
) {
    TRestorableFastRng64 rand(randSeed);
    rand.Advance(10); // reduce correlation between RNGs in different threads
    for (size_t subcandidateIdx = 0; subcandidateIdx < allScores.size(); ++subcandidateIdx) {
        double bestScoreInstance = MINIMAL_SCORE;
        auto& subcandidateInfo = (*subcandidates)[subcandidateIdx];
        const bool isBinaryFeaturesPackEnsemble = subcandidateInfo.SplitEnsemble.IsBinarySplitsPack;

        NCB::TBinaryFeaturesPack binaryFeaturesBinMask;
        if (isBinaryFeaturesPackEnsemble) {
            binaryFeaturesBinMask = perPackMasks[subcandidateInfo.SplitEnsemble.BinarySplitsPack.PackIdx];
        }

        const auto& scores = allScores[subcandidateIdx];
        for (int binFeatureIdx = 0; binFeatureIdx < scores.ysize(); ++binFeatureIdx) {
            if (isBinaryFeaturesPackEnsemble &&
                !(binaryFeaturesBinMask & (NCB::TBinaryFeaturesPack(1) << binFeatureIdx)))
            {
                continue;
            }

            const double score = scores[binFeatureIdx];
            const double scoreInstance = TRandomScore(score, scoreStDev).GetInstance(rand);
            if (scoreInstance > bestScoreInstance) {
                bestScoreInstance = scoreInstance;
                subcandidateInfo.BestScore = TRandomScore(score, scoreStDev);
                subcandidateInfo.BestBinId = binFeatureIdx;
            }
        }
    }
}
