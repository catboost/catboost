#include "train.h"

#include <catboost/private/libs/algo/approx_dimension.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/algo/preprocess.h>
#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>
#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/system_options.h>

#include <catboost/mlx/methods/mlx_boosting.h>
#include <catboost/mlx/targets/pairwise_target.h>
#include <catboost/mlx/targets/pointwise_target.h>
#include <catboost/mlx/train_lib/model_exporter.h>

#include <mlx/mlx.h>

#include <memory>

namespace mx = mlx::core;

namespace NCatboostMlx {

    void TMLXModelTrainer::TrainModel(
        const TTrainModelInternalOptions& internalOptions,
        const NCatboostOptions::TCatBoostOptions& catboostOptions,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        NCB::TTrainingDataProviders trainingData,
        TMaybe<NCB::TPrecomputedOnlineCtrData> precomputedSingleOnlineCtrDataForSingleFold,
        const TLabelConverter& labelConverter,
        ITrainingCallbacks* trainingCallbacks,
        ICustomCallbacks* customCallbacks,
        TMaybe<TFullModel*> initModel,
        THolder<TLearnProgress> initLearnProgress,
        NCB::TDataProviders initModelApplyCompatiblePools,
        NPar::ILocalExecutor* localExecutor,
        const TMaybe<TRestorableFastRng64*> rand,
        TFullModel* dstModel,
        const TVector<TEvalResult*>& evalResultPtrs,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
        THolder<TLearnProgress>* dstLearnProgress
    ) const {
        Y_UNUSED(internalOptions);
        Y_UNUSED(objectiveDescriptor);
        Y_UNUSED(evalMetricDescriptor);
        Y_UNUSED(customCallbacks);
        Y_UNUSED(rand);
        Y_UNUSED(evalResultPtrs);

        CB_ENSURE(trainingData.Test.size() <= 1,
            "Multiple eval sets not yet supported for MLX backend");
        CB_ENSURE(!precomputedSingleOnlineCtrDataForSingleFold,
            "Precomputed online CTR data for MLX backend is not yet supported");
        CB_ENSURE(
            evalResultPtrs.empty() || (evalResultPtrs.size() == trainingData.Test.size()),
            "Need test dataset to evaluate resulting model");
        CB_ENSURE(!initModel && !initLearnProgress,
            "Training continuation for MLX backend is not yet supported");
        Y_UNUSED(initModelApplyCompatiblePools);
        CB_ENSURE_INTERNAL(!dstLearnProgress,
            "Returning learn progress for MLX backend is not yet supported");

        CATBOOST_INFO_LOG << "CatBoost-MLX: Apple Silicon Metal GPU backend initialized" << Endl;

        NCatboostOptions::TCatBoostOptions updatedOptions(catboostOptions);

        const ui32 objectCount = trainingData.Learn->GetObjectCount();
        const ui32 approxDimension = GetApproxDimension(
            updatedOptions,
            labelConverter,
            trainingData.Learn->TargetData->GetTargetDimension()
        );

        CATBOOST_INFO_LOG << "CatBoost-MLX: Training data has " << objectCount
            << " objects, approx dimension = " << approxDimension << Endl;

        // Phase 1: Build TMLXDataSet from trainingData
        auto dataset = NCatboostMlx::BuildMLXDataSet(*trainingData.Learn, approxDimension, localExecutor);
        ui32 numFeatures = static_cast<ui32>(dataset.GetCompressedIndex().GetFeatures().size());
        CATBOOST_INFO_LOG << "CatBoost-MLX: Dataset built with " << numFeatures << " features" << Endl;

        // Phase 2: Extract boosting configuration from catboostOptions
        TBoostingConfig config;
        config.NumIterations = updatedOptions.BoostingOptions->IterationCount.Get();
        config.LearningRate = updatedOptions.BoostingOptions->LearningRate.Get();
        config.MaxDepth = updatedOptions.ObliviousTreeOptions->MaxDepth.Get();
        config.L2RegLambda = updatedOptions.ObliviousTreeOptions->L2Reg.Get();
        config.UseWeights = dataset.HasWeights();
        config.ApproxDimension = approxDimension;

        CATBOOST_INFO_LOG << "CatBoost-MLX: Config: iterations=" << config.NumIterations
            << " lr=" << config.LearningRate
            << " depth=" << config.MaxDepth
            << " l2=" << config.L2RegLambda << Endl;

        // GrowPolicy + MaxLeaves
        {
            const auto catGrowPolicy = updatedOptions.ObliviousTreeOptions->GrowPolicy.Get();
            switch (catGrowPolicy) {
                case EGrowPolicy::SymmetricTree:
                    config.GrowPolicy = NCatboostMlx::EGrowPolicy::SymmetricTree;
                    break;
                case EGrowPolicy::Depthwise:
                    config.GrowPolicy = NCatboostMlx::EGrowPolicy::Depthwise;
                    break;
                case EGrowPolicy::Lossguide:
                    config.GrowPolicy = NCatboostMlx::EGrowPolicy::Lossguide;
                    break;
                default:
                    CB_ENSURE(false,
                        "CatBoost-MLX: Unsupported GrowPolicy: " << catGrowPolicy
                        << ". Supported: SymmetricTree, Depthwise, Lossguide.");
            }
            config.MaxLeaves = updatedOptions.ObliviousTreeOptions->MaxLeaves.Get();
            CATBOOST_INFO_LOG << "CatBoost-MLX: GrowPolicy=" << catGrowPolicy
                << " MaxLeaves=" << config.MaxLeaves << Endl;
        }

        // RandomSeed
        config.RandomSeed = updatedOptions.RandomSeed.Get();
        CATBOOST_INFO_LOG << "CatBoost-MLX: RandomSeed=" << config.RandomSeed << Endl;

        // SubsampleRatio — sourced from BootstrapConfig.TakenFraction.
        // Only Bernoulli and MVS bootstrap types apply a true row subsample fraction;
        // for Bayesian (BaggingTemperature) and No bootstrap we keep the default 1.0.
        {
            const auto& bootstrapConfig = updatedOptions.ObliviousTreeOptions->BootstrapConfig.Get();
            const auto bootstrapType = bootstrapConfig.GetBootstrapType();
            if (bootstrapType == EBootstrapType::Bernoulli || bootstrapType == EBootstrapType::MVS) {
                config.SubsampleRatio = bootstrapConfig.GetTakenFraction();
            }
            CATBOOST_INFO_LOG << "CatBoost-MLX: SubsampleRatio=" << config.SubsampleRatio
                << " (bootstrap=" << bootstrapType << ")" << Endl;
        }

        // ColsampleByTree — sourced from RSM (random subspace method).
        // Rsm == 1.0 means all features are used (no column subsampling).
        config.ColsampleByTree = updatedOptions.ObliviousTreeOptions->Rsm.Get();
        CATBOOST_INFO_LOG << "CatBoost-MLX: ColsampleByTree=" << config.ColsampleByTree << Endl;

        // EarlyStoppingPatience — sourced from the overfitting detector.
        // OverfittingDetectorType::Iter uses IterationsWait as the patience window.
        // Any other type (None, Wilcoxon, IncToDec) leaves early stopping disabled (0).
        {
            const auto& odOptions = updatedOptions.BoostingOptions->OverfittingDetector.Get();
            if (odOptions.OverfittingDetectorType.Get() == EOverfittingDetectorType::Iter) {
                config.EarlyStoppingPatience = static_cast<ui32>(odOptions.IterationsWait.Get());
            }
            CATBOOST_INFO_LOG << "CatBoost-MLX: EarlyStoppingPatience=" << config.EarlyStoppingPatience
                << " (od_type=" << odOptions.OverfittingDetectorType.Get() << ")" << Endl;
        }

        // MetricPeriod — sourced from outputOptions.
        config.MetricPeriod = static_cast<ui32>(outputOptions.GetMetricPeriod());
        CATBOOST_INFO_LOG << "CatBoost-MLX: MetricPeriod=" << config.MetricPeriod << Endl;

        // Phase 3: Create target function
        auto lossFunction = updatedOptions.LossFunctionDescription->GetLossFunction();
        const auto& lossParamsMap = updatedOptions.LossFunctionDescription->GetLossParamsMap();

        std::unique_ptr<IMLXTargetFunc> targetPtr;
        switch (lossFunction) {
            case ELossFunction::RMSE:
                targetPtr = std::make_unique<TRMSETarget>();
                break;
            case ELossFunction::Logloss:
            case ELossFunction::CrossEntropy:
                CB_ENSURE(approxDimension == 1,
                    "CatBoost-MLX: Logloss/CrossEntropy requires approxDimension=1. Got: "
                    << approxDimension);
                targetPtr = std::make_unique<TLoglossTarget>();
                break;
            case ELossFunction::MultiClass:
                CB_ENSURE(approxDimension > 1,
                    "CatBoost-MLX: MultiClass requires approxDimension > 1. Got: "
                    << approxDimension);
                targetPtr = std::make_unique<TMultiClassTarget>(approxDimension + 1);
                break;
            case ELossFunction::MAE:
                CB_ENSURE(approxDimension == 1,
                    "CatBoost-MLX: MAE requires approxDimension=1. Got: "
                    << approxDimension);
                targetPtr = std::make_unique<TMAETarget>();
                break;
            case ELossFunction::Quantile: {
                CB_ENSURE(approxDimension == 1,
                    "CatBoost-MLX: Quantile requires approxDimension=1. Got: "
                    << approxDimension);
                float alpha = NCatboostOptions::GetParamOrDefault(lossParamsMap, TString("alpha"), 0.5f);
                CATBOOST_INFO_LOG << "CatBoost-MLX: Quantile alpha=" << alpha << Endl;
                targetPtr = std::make_unique<TQuantileTarget>(alpha);
                break;
            }
            case ELossFunction::Huber: {
                CB_ENSURE(approxDimension == 1,
                    "CatBoost-MLX: Huber requires approxDimension=1. Got: "
                    << approxDimension);
                CB_ENSURE(lossParamsMap.contains("delta"),
                    "CatBoost-MLX: For " << ELossFunction::Huber
                    << " the delta parameter is mandatory. "
                    << "Specify it as 'Huber:delta=1.0' in CatBoost params "
                    << "or 'huber:1.0' in the csv_train --loss flag.");
                float delta = FromString<float>(lossParamsMap.at("delta"));
                CATBOOST_INFO_LOG << "CatBoost-MLX: Huber delta=" << delta << Endl;
                targetPtr = std::make_unique<THuberTarget>(delta);
                break;
            }
            case ELossFunction::Poisson:
                CB_ENSURE(approxDimension == 1,
                    "CatBoost-MLX: Poisson requires approxDimension=1. Got: "
                    << approxDimension);
                targetPtr = std::make_unique<TPoissonTarget>();
                break;
            case ELossFunction::Tweedie: {
                CB_ENSURE(approxDimension == 1,
                    "CatBoost-MLX: Tweedie requires approxDimension=1. Got: "
                    << approxDimension);
                float p = NCatboostOptions::GetParamOrDefault(lossParamsMap, TString("variance_power"), 1.5f);
                CATBOOST_INFO_LOG << "CatBoost-MLX: Tweedie variance_power=" << p << Endl;
                targetPtr = std::make_unique<TTweedieTarget>(p);
                break;
            }
            case ELossFunction::MAPE:
                CB_ENSURE(approxDimension == 1,
                    "CatBoost-MLX: MAPE requires approxDimension=1. Got: "
                    << approxDimension);
                targetPtr = std::make_unique<TMAPETarget>();
                break;
            case ELossFunction::PairLogit: {
                CB_ENSURE(approxDimension == 1,
                    "CatBoost-MLX: PairLogit requires approxDimension=1. Got: "
                    << approxDimension);
                CB_ENSURE(dataset.HasGroups(),
                    "CatBoost-MLX: PairLogit requires group (query) data. "
                    "Ensure the training pool has a GroupId column.");
                // Pull relevance labels to CPU for pair generation
                const auto& gpuTargets = dataset.GetTargets();
                mx::eval(gpuTargets);
                std::vector<float> cpuTargets(
                    gpuTargets.data<float>(),
                    gpuTargets.data<float>() + objectCount
                );
                targetPtr = std::make_unique<TPairLogitTarget>(
                    cpuTargets,
                    dataset.GetGroupOffsets(),
                    dataset.GetNumGroups()
                );
                CATBOOST_INFO_LOG << "CatBoost-MLX: PairLogit with "
                    << dataset.GetNumGroups() << " groups" << Endl;
                break;
            }
            case ELossFunction::YetiRank: {
                CB_ENSURE(approxDimension == 1,
                    "CatBoost-MLX: YetiRank requires approxDimension=1. Got: "
                    << approxDimension);
                CB_ENSURE(dataset.HasGroups(),
                    "CatBoost-MLX: YetiRank requires group (query) data. "
                    "Ensure the training pool has a GroupId column.");
                const auto& gpuTargets = dataset.GetTargets();
                mx::eval(gpuTargets);
                std::vector<float> cpuTargets(
                    gpuTargets.data<float>(),
                    gpuTargets.data<float>() + objectCount
                );
                const ui32 yetiSeed = config.RandomSeed;
                targetPtr = std::make_unique<TYetiRankTarget>(
                    cpuTargets,
                    dataset.GetGroupOffsets(),
                    dataset.GetNumGroups(),
                    yetiSeed
                );
                CATBOOST_INFO_LOG << "CatBoost-MLX: YetiRank with "
                    << dataset.GetNumGroups() << " groups, seed=" << yetiSeed << Endl;
                break;
            }
            default:
                CB_ENSURE(false,
                    "CatBoost-MLX currently supports RMSE, Logloss, CrossEntropy, MultiClass, "
                    "MAE, Quantile, Huber, Poisson, Tweedie, MAPE, PairLogit, and YetiRank. "
                    "Got: " << lossFunction);
        }

        CATBOOST_INFO_LOG << "CatBoost-MLX: Loss function: " << lossFunction
            << " (approxDim=" << approxDimension << ")" << Endl;

        // Phase 4: Run boosting
        auto result = RunBoosting(
            dataset,
            *targetPtr,
            config,
            trainingCallbacks,
            metricsAndTimeHistory
        );

        CATBOOST_INFO_LOG << "CatBoost-MLX: Training completed with "
            << result.NumIterations << " trees" << Endl;

        // Phase 5: Convert result to TFullModel
        if (dstModel && result.NumIterations > 0) {
            const auto& objectsData = *trainingData.Learn->ObjectsData;
            *dstModel = ConvertToFullModel(
                result,
                *objectsData.GetQuantizedFeaturesInfo(),
                *objectsData.GetFeaturesLayout(),
                dataset.GetCompressedIndex().GetFeatures(),
                dataset.GetCompressedIndex().GetExternalFeatureIndices(),
                approxDimension,
                updatedOptions
            );
            CATBOOST_INFO_LOG << "CatBoost-MLX: Model exported to TFullModel with "
                << result.NumIterations << " trees" << Endl;
        } else if (dstModel) {
            CATBOOST_WARNING_LOG << "CatBoost-MLX: No trees produced, model is empty" << Endl;
        }
        Y_UNUSED(dstModel);
    }

    void TMLXModelTrainer::ModelBasedEval(
        const NCatboostOptions::TCatBoostOptions& /*catboostOptions*/,
        const NCatboostOptions::TOutputFilesOptions& /*outputOptions*/,
        NCB::TTrainingDataProviders /*trainingData*/,
        const TLabelConverter& /*labelConverter*/,
        NPar::ILocalExecutor* /*localExecutor*/
    ) const {
        CB_ENSURE(false, "ModelBasedEval is not yet supported for the MLX backend");
    }

}  // namespace NCatboostMlx

// Register the MLX trainer as the GPU backend on darwin-arm64.
TTrainerFactory::TRegistrator<NCatboostMlx::TMLXModelTrainer> MLXGPURegistrator(ETaskType::GPU);
