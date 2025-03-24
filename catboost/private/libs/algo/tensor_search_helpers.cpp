#include "tensor_search_helpers.h"

#include "calc_score_cache.h"
#include "fold.h"
#include "mvs.h"

#include <catboost/libs/data/objects.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/libs/helpers/distribution_helpers.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/cast.h>
#include <util/generic/maybe.h>
#include <util/generic/xrange.h>


using namespace NCB;


TSplit TCandidateInfo::GetBestSplit(
    const TTrainingDataProviders& data,
    const TFold& fold,
    ui32 oneHotMaxSize
) const {
    const TQuantizedObjectsDataProviderPtr objectsData = [&] () {
        if (SplitEnsemble.IsOnlineEstimated) {
            return fold.GetOnlineEstimatedFeatures().Learn;
        }
        if (SplitEnsemble.IsEstimated) {
            return data.EstimatedObjectsData.Learn;
        }
        return data.Learn->ObjectsData;
    }();

    return GetSplit(BestBinId, *objectsData, oneHotMaxSize);
}

TSplit TCandidateInfo::GetSplit(
    int binId,
    const TQuantizedObjectsDataProvider& objectsData,
    ui32 oneHotMaxSize
) const {
    auto getCandidateType = [&] (EFeatureType featureType) {
        if (SplitEnsemble.IsEstimated) {
            return ESplitType::EstimatedFeature;
        } else if (featureType == EFeatureType::Float) {
            return ESplitType::FloatFeature;
        } else {
            return ESplitType::OneHotFeature;
        }
    };

    switch (SplitEnsemble.Type) {
        case ESplitEnsembleType::OneFeature:
            return TSplit(SplitEnsemble.SplitCandidate, binId);
        case ESplitEnsembleType::BinarySplits:
            {
                TPackedBinaryIndex packedBinaryIndex(SplitEnsemble.BinarySplitsPackRef.PackIdx, binId);
                auto featureInfo = objectsData.GetPackedBinaryFeatureSrcIndex(packedBinaryIndex);
                TSplitCandidate splitCandidate;

                splitCandidate.Type = getCandidateType(featureInfo.FeatureType);
                splitCandidate.FeatureIdx = featureInfo.FeatureIdx;
                splitCandidate.IsOnlineEstimatedFeature = SplitEnsemble.IsOnlineEstimated;

                return TSplit(
                    std::move(splitCandidate),
                    (featureInfo.FeatureType == EFeatureType::Float) ? 0 : 1);
            }
        case ESplitEnsembleType::ExclusiveBundle:
            {
                const auto bundleIdx = SplitEnsemble.ExclusiveFeaturesBundleRef.BundleIdx;
                const auto& bundleMetaData = objectsData.GetExclusiveFeatureBundlesMetaData()[bundleIdx];

                ui32 binFeatureOffset = 0;
                for (const auto& bundlePart : bundleMetaData.Parts) {
                    if (!UseForCalcScores(bundlePart, oneHotMaxSize)) {
                        continue;
                    }

                    const auto binFeatureSize = (bundlePart.FeatureType == EFeatureType::Float) ?
                        bundlePart.Bounds.GetSize() :
                        bundlePart.Bounds.GetSize() + 1;

                    const auto binInBundlePart = binId - binFeatureOffset;

                    if (binInBundlePart < binFeatureSize) {
                        TSplitCandidate splitCandidate;
                        splitCandidate.Type = getCandidateType(bundlePart.FeatureType);
                        splitCandidate.FeatureIdx = bundlePart.FeatureIdx;

                        return TSplit(std::move(splitCandidate), binInBundlePart);
                    }

                    binFeatureOffset += binFeatureSize;
                }
                CB_ENSURE(false, "This should be unreachable");
                // keep compiler happy
                return TSplit();
            }
        case ESplitEnsembleType::FeaturesGroup:
            {
                const auto groupIdx = SplitEnsemble.FeaturesGroupRef.GroupIdx;
                const auto& groupMetaData = objectsData.GetFeaturesGroupMetaData(groupIdx);

                ui32 splitIdxOffset = 0;
                for (const auto& part : groupMetaData.Parts) {
                    ui32 splitIdxInPart = binId - splitIdxOffset;
                    if (splitIdxInPart < part.BucketCount - 1) {
                        TSplitCandidate splitCandidate;
                        splitCandidate.Type = getCandidateType(part.FeatureType);
                        splitCandidate.FeatureIdx = part.FeatureIdx;

                        return TSplit(std::move(splitCandidate), splitIdxInPart);
                    }

                    splitIdxOffset += part.BucketCount - 1;
                }
                CB_ENSURE(false, "This should be unreachable");
                // keep compiler happy
                return TSplit();
            }
    }
    Y_UNREACHABLE();
}


THolder<IDerCalcer> BuildError(
    const NCatboostOptions::TCatBoostOptions& params,
    const TMaybe<TCustomObjectiveDescriptor>& descriptor
) {
    const bool isStoreExpApprox = IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction());
    switch (params.LossFunctionDescription->GetLossFunction()) {
        case ELossFunction::SurvivalAft: {
            const auto& lossParams = params.LossFunctionDescription->GetLossParamsMap();
            for (const auto& param : lossParams) {
                CB_ENSURE(
                    param.first == "dist" || param.first == "scale",
                    "Invalid loss description" << ToString(params.LossFunctionDescription.Get()));
            }
            std::unique_ptr<IDistribution> distribution;
            if (lossParams.contains("dist")) {
                switch (FromString<EDistributionType>(lossParams.at("dist"))) {
                    case EDistributionType::Extreme:
                        distribution = std::make_unique<TExtremeDistribution>();
                        break;
                    case EDistributionType::Logistic:
                        distribution = std::make_unique<TLogisticDistribution>();
                        break;
                    case EDistributionType::Normal:
                        distribution = std::make_unique<TNormalDistribution>();
                        break;
                    default:
                        CB_ENSURE(false, "Unsupported distribution type " << lossParams.at("dist"));
               }
            } else {
               distribution = std::make_unique<TNormalDistribution>();
            }
            double scale = lossParams.contains("scale") ? FromString<double>(lossParams.at("scale")) : 1;
            return MakeHolder<TSurvivalAftError>(std::move(distribution), scale);
        }
        case ELossFunction::MultiRMSE:
            return MakeHolder<TMultiRMSEError>();
        case ELossFunction::MultiRMSEWithMissingValues:
            return MakeHolder<TMultiRMSEErrorWithMissingValues>();
        case ELossFunction::RMSEWithUncertainty: {
            return MakeHolder<TRMSEWithUncertaintyError>();
        }
        case ELossFunction::Logloss:
        case ELossFunction::CrossEntropy:
            return MakeHolder<TCrossEntropyError>(isStoreExpApprox);
        case ELossFunction::RMSE:
            return MakeHolder<TRMSEError>(isStoreExpApprox);
        case ELossFunction::Cox:
            return MakeHolder<TCoxError>(isStoreExpApprox);
        case ELossFunction::MAE:
        case ELossFunction::Quantile: {
            const auto& lossParams = params.LossFunctionDescription->GetLossParamsMap();
            const auto badParam = FindIf(lossParams, [] (const auto& param) { return !EqualToOneOf(param.first, "alpha", "delta"); });
            CB_ENSURE(badParam == lossParams.end(), "Invalid loss description " << ToString(badParam->first));
            double alpha = lossParams.contains("alpha") ? FromString<float>(lossParams.at("alpha")) : 0.5;
            double delta = lossParams.contains("delta") ? FromString<float>(lossParams.at("delta")) : 1e-6;
            return MakeHolder<TQuantileError>(alpha, delta, isStoreExpApprox);
        }
        case ELossFunction::GroupQuantile: {
            const auto& lossParams = params.LossFunctionDescription->GetLossParamsMap();
            const auto badParam = FindIf(lossParams, [] (const auto& param) { return !EqualToOneOf(param.first, "alpha", "delta"); });
            CB_ENSURE(badParam == lossParams.end(), "Invalid loss description " << ToString(badParam->first));
            double alpha = lossParams.contains("alpha") ? FromString<float>(lossParams.at("alpha")) : 0.5;
            double delta = lossParams.contains("delta") ? FromString<float>(lossParams.at("delta")) : 1e-6;
            return MakeHolder<TGroupQuantileError>(alpha, delta, isStoreExpApprox);
        }
        case ELossFunction::MultiQuantile: {
            const auto& lossParams = params.LossFunctionDescription->GetLossParamsMap();
            const auto badParam = FindIf(lossParams, [] (const auto& param) { return !EqualToOneOf(param.first, "alpha", "delta"); });
            CB_ENSURE(badParam == lossParams.end(), "Invalid loss description " << ToString(badParam->first));
            const auto alpha = NCatboostOptions::GetAlphaMultiQuantile(lossParams);
            CB_ENSURE(alpha.size() >= 2, "Parameter alpha should contain at least two quantiles separated by comma");
            double delta = lossParams.contains("delta") ? FromString<float>(lossParams.at("delta")) : 1e-6;
            return MakeHolder<TMultiQuantileError>(alpha, delta, isStoreExpApprox);
        }
        case ELossFunction::Expectile: {
            const auto& lossParams = params.LossFunctionDescription->GetLossParamsMap();
            if (lossParams.empty()) {
                return MakeHolder<TExpectileError>(isStoreExpApprox);
            } else {
                CB_ENSURE(lossParams.begin()->first == "alpha", "Invalid loss description " << ToString(params.LossFunctionDescription.Get()));
                return MakeHolder<TExpectileError>(FromString<float>(lossParams.at("alpha")), isStoreExpApprox);
            }
        }
        case ELossFunction::LogLinQuantile: {
            const auto& lossParams = params.LossFunctionDescription->GetLossParamsMap();
            if (lossParams.empty()) {
                return MakeHolder<TLogLinQuantileError>(isStoreExpApprox);
            } else {
                CB_ENSURE(
                    lossParams.begin()->first == "alpha",
                    "Invalid loss description " << ToString(params.LossFunctionDescription.Get()));
                return MakeHolder<TLogLinQuantileError>(
                    FromString<float>(lossParams.at("alpha")),
                    isStoreExpApprox);
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
        case ELossFunction::MultiLogloss:
        case ELossFunction::MultiCrossEntropy:
            return MakeHolder<TMultiCrossEntropyError>();
        case ELossFunction::PairLogit:
            return MakeHolder<TPairLogitError>(isStoreExpApprox);
        case ELossFunction::PairLogitPairwise:
            return MakeHolder<TPairLogitError>(isStoreExpApprox);
        case ELossFunction::QueryRMSE:
            return MakeHolder<TQueryRmseError>(isStoreExpApprox);
        case ELossFunction::QuerySoftMax: {
            const auto& lossFunctionDescription = params.LossFunctionDescription;
            const auto& lossParams = lossFunctionDescription->GetLossParamsMap();
            for (const auto& [param, value] : lossParams) {
                Y_UNUSED(value);
                CB_ENSURE(
                    param == "lambda" || param == "beta",
                    "Invalid loss description" << ToString(lossFunctionDescription.Get()));
            }

            const double lambdaReg = NCatboostOptions::GetQuerySoftMaxLambdaReg(lossFunctionDescription);
            const double beta = NCatboostOptions::GetQuerySoftMaxBeta(lossFunctionDescription);
            return MakeHolder<TQuerySoftMaxError>(lambdaReg, beta, isStoreExpApprox);
        }
        case ELossFunction::YetiRank:
            return MakeHolder<TPairLogitError>(isStoreExpApprox);
        case ELossFunction::YetiRankPairwise:
            return MakeHolder<TPairLogitError>(isStoreExpApprox);
        case ELossFunction::Lq:
            return MakeHolder<TLqError>(
                NCatboostOptions::GetLqParam(params.LossFunctionDescription),
                isStoreExpApprox);
        case ELossFunction::StochasticFilter: {
            double sigma = NCatboostOptions::GetStochasticFilterSigma(params.LossFunctionDescription);
            int numEstimations = NCatboostOptions::GetStochasticFilterNumEstimations(
                params.LossFunctionDescription);
            return MakeHolder<TStochasticFilterError>(sigma, numEstimations, isStoreExpApprox);
        }
        case ELossFunction::LambdaMart: {
            const auto& lossParams = params.LossFunctionDescription->GetLossParamsMap();
            const ELossFunction targetMetric = lossParams.contains("metric") ? FromString<ELossFunction>(lossParams.at("metric")) : ELossFunction::NDCG;
            const double sigma = NCatboostOptions::GetParamOrDefault(lossParams, "sigma", 1.0);
            const bool norm = NCatboostOptions::GetParamOrDefault(lossParams, "norm", true);
            return MakeHolder<TLambdaMartError>(targetMetric, lossParams, sigma, norm);
        }
        case ELossFunction::StochasticRank: {
            const auto& lossParams = params.LossFunctionDescription->GetLossParamsMap();
            CB_ENSURE(lossParams.contains("metric"), "StochasticRank requires metric param");
            const ELossFunction targetMetric = FromString<ELossFunction>(lossParams.at("metric"));
            const double sigma = NCatboostOptions::GetParamOrDefault(lossParams, "sigma", 1.0);
            const size_t numEstimations = NCatboostOptions::GetParamOrDefault(lossParams, "num_estimations", size_t(1));
            const double mu = NCatboostOptions::GetParamOrDefault(lossParams, "mu", 0.0);
            const double nu = NCatboostOptions::GetParamOrDefault(lossParams, "nu", 0.01);
            const double defaultLambda = targetMetric == ELossFunction::FilteredDCG ? 0.0 : 1.0;
            const double lambda = NCatboostOptions::GetParamOrDefault(lossParams, "lambda", defaultLambda);
            return MakeHolder<TStochasticRankError>(targetMetric, lossParams, sigma, numEstimations, mu, nu, lambda);
        }
        case ELossFunction::PythonUserDefinedPerObject:
            return MakeHolder<TCustomError>(params, descriptor);
        case ELossFunction::PythonUserDefinedMultiTarget:
            return MakeHolder<TMultiTargetCustomError>(params, descriptor);
        case ELossFunction::UserPerObjMetric:
            return MakeHolder<TUserDefinedPerObjectError>(
                    params.LossFunctionDescription->GetLossParamsMap(),
                isStoreExpApprox);
        case ELossFunction::UserQuerywiseMetric:
            return MakeHolder<TUserDefinedQuerywiseError>(
                    params.LossFunctionDescription->GetLossParamsMap(),
                isStoreExpApprox);
        case ELossFunction::Huber:
            return MakeHolder<THuberError>(NCatboostOptions::GetHuberParam(params.LossFunctionDescription), isStoreExpApprox);
        case ELossFunction::Tweedie:
            return MakeHolder<TTweedieError>(
                NCatboostOptions::GetTweedieParam(params.LossFunctionDescription),
                isStoreExpApprox);
        case ELossFunction::Focal:
            return MakeHolder<TFocalError>(
                NCatboostOptions::GetFocalParamA(params.LossFunctionDescription),
                NCatboostOptions::GetFocalParamG(params.LossFunctionDescription),
                isStoreExpApprox);
        case ELossFunction::LogCosh:
            return MakeHolder<TLogCoshError>(isStoreExpApprox);
        default:
            CB_ENSURE(false, "provided error function is not supported");
    }
}

static float GenerateBayessianWeight(float baggingTemperature, TRestorableFastRng64& rand) {
    const float w = -FastLogf(rand.GenRandReal1() + 1e-100);
    return powf(w, baggingTemperature);
}

static void GenerateRandomWeights(
    int learnSampleCount,
    float baggingTemperature,
    ESamplingUnit samplingUnit,
    NPar::ILocalExecutor* localExecutor,
    TRestorableFastRng64* rand,
    TFold* fold
) {
    if (baggingTemperature == 0) {
        Fill(fold->SampleWeights.begin(), fold->SampleWeights.end(), 1);
        return;
    }

    const int groupCount = fold->LearnQueriesInfo.ysize();
    const ui64 randSeed = rand->GenRand();
    const int sampleCount = (samplingUnit == ESamplingUnit::Group) ? groupCount : learnSampleCount;

    NPar::ILocalExecutor::TExecRangeParams blockParams(0, sampleCount);
    blockParams.SetBlockSize(1000);
    localExecutor->ExecRange(
        [&](int blockIdx) {
            TRestorableFastRng64 rand(randSeed + blockIdx);
            rand.Advance(10); // reduce correlation between RNGs in different threads
            float* sampleWeightsData = fold->SampleWeights.data();
            NPar::TLocalExecutor::BlockedLoopBody(
                blockParams,
                [=,&rand](int i) {
                const float w = GenerateBayessianWeight(baggingTemperature, rand);
                    if (samplingUnit == ESamplingUnit::Object) {
                        sampleWeightsData[i] = w;
                    } else {
                        ui32 begin = fold->LearnQueriesInfo[i].Begin;
                        ui32 end = fold->LearnQueriesInfo[i].End;
                        Fill(sampleWeightsData + begin, sampleWeightsData + end, w);
                    }
                })(blockIdx);
        },
        0,
        blockParams.GetBlockCount(),
        NPar::TLocalExecutor::WAIT_COMPLETE);
}

static void GenerateBayesianWeightsForPairs(
    float baggingTemperature,
    ESamplingUnit samplingUnit,
    NPar::ILocalExecutor* localExecutor,
    TRestorableFastRng64* rand,
    TFold* fold
) {
    if (baggingTemperature == 0.0f) {
        return;
    }
    const ui64 randSeed = rand->GenRand();
    NPar::ILocalExecutor::TExecRangeParams blockParams(0, fold->LearnQueriesInfo.ysize());
    blockParams.SetBlockSize(1000);
    localExecutor->ExecRange(
        [&](int blockIdx) {
            TRestorableFastRng64 rand(randSeed + blockIdx);
            rand.Advance(10); // reduce correlation between RNGs in different threads
            NPar::TLocalExecutor::BlockedLoopBody(
                blockParams,
                [&](int i) {
                    const float wGroup = GenerateBayessianWeight(baggingTemperature, rand);
                    for (auto& competitors : fold->LearnQueriesInfo[i].Competitors) {
                        for (auto& competitor : competitors) {
                            float w = (samplingUnit == ESamplingUnit::Group) ?
                                wGroup : GenerateBayessianWeight(baggingTemperature, rand);
                            competitor.SampleWeight = competitor.Weight * w;
                        }
                    }
                })(blockIdx);
        },
        0,
        blockParams.GetBlockCount(),
        NPar::TLocalExecutor::WAIT_COMPLETE);
}

static void GenerateBernoulliWeightsForPairs(
    float takenFraction,
    ESamplingUnit samplingUnit,
    NPar::ILocalExecutor* localExecutor,
    TRestorableFastRng64* rand,
    TFold* fold
) {
    if (takenFraction == 1.0f) {
        return;
    }
    const ui64 randSeed = rand->GenRand();
    NPar::ILocalExecutor::TExecRangeParams blockParams(0, fold->LearnQueriesInfo.ysize());
    blockParams.SetBlockSize(1000);
    localExecutor->ExecRange(
        [&](int blockIdx) {
            TRestorableFastRng64 rand(randSeed + blockIdx);
            rand.Advance(10); // reduce correlation between RNGs in different threads
            NPar::TLocalExecutor::BlockedLoopBody(
                blockParams,
                [&](int i) {
                    const double wGroup = rand.GenRandReal1();
                    for (auto& competitors : fold->LearnQueriesInfo[i].Competitors) {
                        for (auto& competitor : competitors) {
                            double w = (samplingUnit == ESamplingUnit::Group) ? wGroup : rand.GenRandReal1();
                            if (w < takenFraction) {
                                competitor.SampleWeight = competitor.Weight;
                            } else {
                                competitor.SampleWeight = 0.0f;
                            }
                        }
                    }
                })(blockIdx);
        },
        0,
        blockParams.GetBlockCount(),
        NPar::TLocalExecutor::WAIT_COMPLETE);
}

static void CalcWeightedData(
    int learnSampleCount,
    EBoostingType boostingType,
    NPar::ILocalExecutor* localExecutor,
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
            localExecutor->ExecRange(
                [=](int z) {
                    samplePairwiseWeightsData[z] = pairwiseWeightsData[z] * sampleWeightsData[z];
                },
                NPar::ILocalExecutor::TExecRangeParams(begin, bt.TailFinish).SetBlockSize(4000),
                NPar::TLocalExecutor::WAIT_COMPLETE);
        }
        for (int dim = 0; dim < approxDimension; ++dim) {
            const double* weightedDerivativesData = bt.WeightedDerivatives[dim].data();
            double* sampleWeightedDerivativesData = bt.SampleWeightedDerivatives[dim].data();
            localExecutor->ExecRange(
                [=](int z) {
                    sampleWeightedDerivativesData[z] = weightedDerivativesData[z] * sampleWeightsData[z];
                },
                NPar::ILocalExecutor::TExecRangeParams(begin, bt.TailFinish).SetBlockSize(4000),
                NPar::TLocalExecutor::WAIT_COMPLETE);
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
    bool hasOfflineEstimatedFeatures,
    TConstArrayRef<TIndexType> indices,
    const TVector<TVector<TVector<double>>>& leafValues,
    TFold* fold,
    TCalcScoreFold* sampledDocs,
    NPar::ILocalExecutor* localExecutor,
    TRestorableFastRng64* rand,
    bool shouldSortByLeaf,
    ui32 leavesCount
) {
    const int learnSampleCount = SafeIntegerCast<int>(indices.size());
    const EBootstrapType bootstrapType = params.ObliviousTreeOptions->BootstrapConfig->GetBootstrapType();
    const EBoostingType boostingType = params.BoostingOptions->BoostingType;
    const ESamplingUnit samplingUnit = params.ObliviousTreeOptions->BootstrapConfig->GetSamplingUnit();
    const float baggingTemperature = params.ObliviousTreeOptions->BootstrapConfig->GetBaggingTemperature();
    const float takenFraction = params.ObliviousTreeOptions->BootstrapConfig->GetTakenFraction();
    const bool isPairwiseScoring = IsPairwiseScoring(params.LossFunctionDescription->GetLossFunction());
    const TMaybe<float> mvsReg = params.ObliviousTreeOptions->BootstrapConfig->GetMvsReg();
    bool performRandomChoice = true;
    if (bootstrapType != EBootstrapType::No && samplingUnit == ESamplingUnit::Group) {
        CB_ENSURE(!fold->LearnQueriesInfo.empty(), "No groups in dataset. Please disable sampling or use per object sampling");
    }
    switch (bootstrapType) {
        case EBootstrapType::Bernoulli:
            if (isPairwiseScoring) {
                GenerateBernoulliWeightsForPairs(takenFraction, samplingUnit, localExecutor, rand, fold);
            } else {
                Fill(fold->SampleWeights.begin(), fold->SampleWeights.end(), 1);
            }
            break;
        case EBootstrapType::Bayesian:
            if (isPairwiseScoring) {
                GenerateBayesianWeightsForPairs(baggingTemperature, samplingUnit, localExecutor, rand, fold);
            } else {
                GenerateRandomWeights(
                    learnSampleCount,
                    baggingTemperature,
                    samplingUnit,
                    localExecutor,
                    rand,
                    fold);
            }
            break;
        case EBootstrapType::MVS:
            CB_ENSURE(
                samplingUnit != ESamplingUnit::Group,
                "MVS bootstrap is not implemented for groupwise sampling (sampling_unit=Group)"
            );
            if (!isPairwiseScoring) {
                performRandomChoice = false;
                TMvsSampler sampler(learnSampleCount, takenFraction, mvsReg);
                sampler.GenSampleWeights(boostingType, leafValues, rand, localExecutor, fold);
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
    sampledDocs->Sample(
        *fold,
        samplingUnit,
        hasOfflineEstimatedFeatures,
        indices,
        rand,
        localExecutor,
        performRandomChoice,
        shouldSortByLeaf,
        leavesCount);
    CB_ENSURE(sampledDocs->GetDocCount() > 0, "Too few sampling units (subsample=" << takenFraction
        << ", bootstrap_type=" << bootstrapType << "): please increase sampling rate or disable sampling");
}

void CalcWeightedDerivatives(
    const IDerCalcer& error,
    int bodyTailIdx,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    TFold* takenFold,
    NPar::ILocalExecutor* localExecutor
) {
    TFold::TBodyTail& bt = takenFold->BodyTailArr[bodyTailIdx];
    const TVector<TVector<double>>& approx = bt.Approx;
    const TVector<float>& target = takenFold->LearnTarget[0];
    const TVector<float>& weight = takenFold->GetLearnWeights();
    TVector<TVector<double>>* weightedDerivatives = &bt.WeightedDerivatives;

    if (error.GetErrorType() == EErrorType::QuerywiseError ||
        error.GetErrorType() == EErrorType::PairwiseError)
    {
        TVector<TQueryInfo> recalculatedQueriesInfo;
        const bool shouldGenerateYetiRankPairs = IsYetiRankLossFunction(
            params.LossFunctionDescription->GetLossFunction());
        if (shouldGenerateYetiRankPairs) {
            YetiRankRecalculation(
                *takenFold,
                bt,
                params,
                randomSeed,
                localExecutor,
                &recalculatedQueriesInfo,
                &bt.PairwiseWeights);
        }
        const TVector<TQueryInfo>& queriesInfo
            = shouldGenerateYetiRankPairs ? recalculatedQueriesInfo : takenFold->LearnQueriesInfo;

        const int tailQueryFinish = bt.TailQueryFinish;
        TVector<TDers> ders((*weightedDerivatives)[0].ysize());
        error.CalcDersForQueries(
            0,
            tailQueryFinish,
            approx[0],
            target,
            weight,
            queriesInfo,
            ders,
            randomSeed,
            localExecutor);
        for (int docId = 0; docId < ders.ysize(); ++docId) {
            (*weightedDerivatives)[0][docId] = ders[docId].Der1;
        }
        if (params.LossFunctionDescription->GetLossFunction() == ELossFunction::YetiRankPairwise) {
            // In case of YetiRankPairwise loss function we need to store generated pairs for tree structure
            //  building.
            Y_ASSERT(takenFold->BodyTailArr.size() == 1);
            takenFold->LearnQueriesInfo.swap(recalculatedQueriesInfo);
        }
    } else {
        const int tailFinish = bt.TailFinish;
        const int approxDimension = approx.ysize();
        NPar::ILocalExecutor::TExecRangeParams blockParams(0, tailFinish);
        blockParams.SetBlockSize(1000);

        Y_ASSERT(error.GetErrorType() == EErrorType::PerObjectError);
        if (const auto multiError = dynamic_cast<const TMultiDerCalcer*>(&error)) {
            const auto& multiTarget = takenFold->LearnTarget;
            localExecutor->ExecRangeWithThrow(
                [&](int blockId) {
                    TVector<double> curApprox(approxDimension);
                    TVector<float> curTarget(multiTarget.size());
                    TVector<double> curDelta(approxDimension);
                    NPar::TLocalExecutor::BlockedLoopBody(
                        blockParams,
                        [&](int docId) {
                            for (auto dim : xrange(approxDimension)) {
                                curApprox[dim] = approx[dim][docId];
                            }
                            for (auto dim : xrange(multiTarget.size())) {
                                curTarget[dim] = multiTarget[dim][docId];
                            }
                            multiError->CalcDers(
                                curApprox,
                                curTarget,
                                weight.empty() ? 1 : weight[docId],
                                &curDelta,
                                nullptr
                            );
                            for (int dim = 0; dim < approxDimension; ++dim) {
                                (*weightedDerivatives)[dim][docId] = curDelta[dim];
                            }
                        })(blockId);
                },
                0,
                blockParams.GetBlockCount(),
                NPar::TLocalExecutor::WAIT_COMPLETE);
        } else if (approxDimension == 1) {
            if (dynamic_cast<const TCoxError*>(&error) == nullptr) {
                localExecutor->ExecRangeWithThrow(
                    [&](int blockId) {
                        const int blockOffset = blockId * blockParams.GetBlockSize();
                        error.CalcFirstDerRange(
                            blockOffset,
                            Min<int>(blockParams.GetBlockSize(), tailFinish - blockOffset),
                            approx[0].data(),
                            nullptr, // no approx deltas
                            target.data(),
                            weight.data(),
                            (*weightedDerivatives)[0].data());
                    },
                    0,
                    blockParams.GetBlockCount(),
                    NPar::TLocalExecutor::WAIT_COMPLETE);
            } else {
                error.CalcFirstDerRange(
                    /*start*/ 0,
                    /*count*/ tailFinish,
                    /*approx*/ approx[0].data(),
                    /*approx deltas*/ nullptr,
                    /*targets*/ target.data(),
                    /*weights*/ weight.data(),
                    /*first ders*/ (*weightedDerivatives)[0].data());
            }
        } else {
            localExecutor->ExecRangeWithThrow(
                [&](int blockId) {
                    TVector<double> curApprox(approxDimension);
                    TVector<double> curDelta(approxDimension);
                    NPar::TLocalExecutor::BlockedLoopBody(
                        blockParams,
                        [&](int z) {
                            for (int dim = 0; dim < approxDimension; ++dim) {
                                curApprox[dim] = approx[dim][z];
                            }
                            error.CalcDersMulti(
                                curApprox,
                                target[z],
                                weight.empty() ? 1 : weight[z],
                                &curDelta,
                                nullptr);
                            for (int dim = 0; dim < approxDimension; ++dim) {
                                (*weightedDerivatives)[dim][z] = curDelta[dim];
                            }
                        })(blockId);
                },
                0,
                blockParams.GetBlockCount(),
                NPar::TLocalExecutor::WAIT_COMPLETE);
        }
    }
}

void SetBestScore(
    ui64 randSeed,
    const TVector<TVector<double>>& allScores,
    ERandomScoreDistribution scoreDistribution,
    double scoreStDev,
    const TCandidatesContext& candidatesContext,
    TVector<TCandidateInfo>* subcandidates
) {
    TRestorableFastRng64 rand(randSeed);
    rand.Advance(10); // reduce correlation between RNGs in different threads
    for (size_t subcandidateIdx = 0; subcandidateIdx < allScores.size(); ++subcandidateIdx) {
        double bestScoreInstance = MINIMAL_SCORE;
        auto& subcandidateInfo = (*subcandidates)[subcandidateIdx];
        const auto& scores = allScores[subcandidateIdx];

        auto scoreUpdateFunction = [&] (auto binFeatureIdx) {
            const double scoreWoNoise = scores[binFeatureIdx];
            TRandomScore randomScore(scoreDistribution, scoreWoNoise, scoreStDev);
            const double scoreInstance = randomScore.GetInstance(rand);
            if (scoreInstance > bestScoreInstance) {
                bestScoreInstance = scoreInstance;
                subcandidateInfo.BestScore = std::move(randomScore);
                subcandidateInfo.BestBinId = binFeatureIdx;
            }
        };

        switch (subcandidateInfo.SplitEnsemble.Type) {
            case ESplitEnsembleType::OneFeature:
                for (auto binFeatureIdx : xrange(scores.ysize())) {
                    scoreUpdateFunction(binFeatureIdx);
                }
                break;
            case ESplitEnsembleType::BinarySplits:
                {
                    const auto packIdx = subcandidateInfo.SplitEnsemble.BinarySplitsPackRef.PackIdx;
                    const auto binaryFeaturesBinMask = candidatesContext.PerBinaryPackMasks[packIdx];
                    for (auto binFeatureIdx : xrange(scores.ysize())) {
                        if (binaryFeaturesBinMask & (NCB::TBinaryFeaturesPack(1) << binFeatureIdx)) {
                            scoreUpdateFunction(binFeatureIdx);
                        }
                    }
                }
                break;
            case ESplitEnsembleType::ExclusiveBundle:
                {
                    const auto bundleIdx = subcandidateInfo.SplitEnsemble.ExclusiveFeaturesBundleRef.BundleIdx;

                    const THashSet<ui32> selectedFeaturesInBundle(
                        candidatesContext.SelectedFeaturesInBundles[bundleIdx].begin(),
                        candidatesContext.SelectedFeaturesInBundles[bundleIdx].end());

                    ui32 binFeatureOffset = 0;
                    const auto& bundleParts = candidatesContext.BundlesMetaData[bundleIdx].Parts;
                    for (auto bundlePartIdx : xrange(bundleParts.size())) {
                        const auto& bundlePart = bundleParts[bundlePartIdx];

                        if (!UseForCalcScores(bundlePart, candidatesContext.OneHotMaxSize)) {
                            continue;
                        }

                        const auto binFeatureSize = (bundlePart.FeatureType == EFeatureType::Float) ?
                            bundlePart.Bounds.GetSize() :
                            bundlePart.Bounds.GetSize() + 1;

                        if (selectedFeaturesInBundle.contains(bundlePartIdx)) {
                            for (auto binFeatureIdx :
                                 xrange(binFeatureOffset, binFeatureOffset + binFeatureSize))
                            {
                                scoreUpdateFunction(binFeatureIdx);
                            }
                        }

                        binFeatureOffset += binFeatureSize;
                    }
                }
                break;
            case ESplitEnsembleType::FeaturesGroup:
                {
                    const auto groupIdx = subcandidateInfo.SplitEnsemble.FeaturesGroupRef.GroupIdx;

                    const THashSet<ui32> selectedFeaturesInGroup(
                        candidatesContext.SelectedFeaturesInGroups[groupIdx].begin(),
                        candidatesContext.SelectedFeaturesInGroups[groupIdx].end());

                    ui32 splitIdxOffset = 0;
                    const auto& groupParts = candidatesContext.FeaturesGroupsMetaData[groupIdx].Parts;
                    for (auto groupPartIdx : xrange(groupParts.size())) {
                        const auto& groupPart = groupParts[groupPartIdx];

                        const auto splitsCount = groupPart.BucketCount - 1;

                        if (selectedFeaturesInGroup.contains(groupPartIdx)) {
                            for (auto splitIdx : xrange(splitIdxOffset, splitIdxOffset + splitsCount)) {
                                scoreUpdateFunction(splitIdx);
                            }
                        }

                        splitIdxOffset += splitsCount;
                    }
                }
                break;
        }
    }
}
