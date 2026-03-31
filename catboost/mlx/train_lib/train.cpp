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
        Y_UNUSED(trainingCallbacks);
        Y_UNUSED(customCallbacks);
        Y_UNUSED(rand);
        Y_UNUSED(dstModel);
        Y_UNUSED(evalResultPtrs);
        Y_UNUSED(metricsAndTimeHistory);

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

        // TODO(Phase 6): Metal kernel dispatch via mx::fast::metal_kernel()
        // Example (once ready):
        //   auto histKernel = mx::fast::metal_kernel(
        //       "histogram_one_byte_features",
        //       {"compressedIndex", "stats", "partOffsets", ...},
        //       {"histogram"},
        //       metalSource,
        //       /*header=*/"",
        //       /*ensure_row_contiguous=*/true,
        //       /*atomic_outputs=*/true);
        //   auto results = histKernel({...}, {outputShape}, {mx::float32},
        //       grid_dims, threadgroup_dims, ...);

        // TODO(Phase 2-5): Initialize histogram, scoring, leaf, and target kernels
        // TODO(Phase 6): Run boosting loop

        CB_ENSURE(false,
            "CatBoost-MLX training loop not yet implemented. "
            "Backend skeleton registered successfully — Metal kernel implementation in progress.");
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
// This is safe because the CUDA TGPUModelTrainer registration (in catboost/cuda/train_lib/train.cpp)
// only compiles on CUDA-enabled builds, which never target macOS ARM.
TTrainerFactory::TRegistrator<NCatboostMlx::TMLXModelTrainer> MLXGPURegistrator(ETaskType::GPU);
