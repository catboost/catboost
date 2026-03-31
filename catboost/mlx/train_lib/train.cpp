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
#include <catboost/mlx/targets/pointwise_target.h>
#include <catboost/mlx/train_lib/model_exporter.h>

#include <mlx/mlx.h>

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
        Y_UNUSED(outputOptions);
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

        CATBOOST_INFO_LOG << "CatBoost-MLX: Config: iterations=" << config.NumIterations
            << " lr=" << config.LearningRate
            << " depth=" << config.MaxDepth
            << " l2=" << config.L2RegLambda << Endl;

        // Phase 3: Create target function
        auto lossFunction = updatedOptions.LossFunctionDescription->GetLossFunction();
        CB_ENSURE(lossFunction == ELossFunction::RMSE,
            "CatBoost-MLX currently only supports RMSE loss. Got: " << lossFunction);
        TRMSETarget target;

        // Phase 4: Run boosting
        auto result = RunBoosting(
            dataset,
            target,
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
