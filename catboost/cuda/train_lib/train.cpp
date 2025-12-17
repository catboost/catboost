#include "train.h"

#include <catboost/cuda/cpu_compatibility_helpers/model_converter.h>
#include <catboost/cuda/ctrs/prior_estimator.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_lib/devices_provider.h>
#include <catboost/cuda/cuda_lib/memcpy_tracker.h>
#include <catboost/cuda/data/gpu_input_provider.h>
#include <catboost/cuda/gpu_data/pinned_memory_estimation.h>
#include <catboost/cuda/gpu_data/kernel/binarize.cuh>
#include <catboost/cuda/gpu_data/kernel/gpu_input_utils.cuh>

#include <catboost/private/libs/algo/approx_dimension.h>
#include <catboost/private/libs/algo/full_model_saver.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/algo/online_ctr.h>
#include <catboost/private/libs/algo/preprocess.h>
#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>
#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/fstr/output_fstr.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/memory_utils.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/metric_options.h>
#include <catboost/private/libs/options/system_options.h>
#include <catboost/private/libs/quantization/grid_creator.h>
#include <catboost/private/libs/quantization/utils.h>
#include <catboost/libs/train_lib/dir_helper.h>
#include <catboost/libs/train_lib/options_helper.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/cpp/json/json_prettifier.h>
#include <library/cpp/json/json_value.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/folder/path.h>
#include <util/generic/algorithm.h>
#include <util/generic/scope.h>
#include <util/system/compiler.h>
#include <util/system/guard.h>
#include <util/system/info.h>
#include <util/system/spinlock.h>
#include <util/system/yassert.h>
#include <catboost/cuda/models/compact_model.h>

#include <cuda_runtime_api.h>

#include <cmath>

using namespace NCB;

namespace NCatboostCuda {
    inline void UpdatePinnedMemorySizeOption(const NCB::TTrainingDataProvider& learn,
                                             const NCB::TTrainingDataProvider* test,
                                             const TBinarizedFeaturesManager& featuresManager,
                                             NCatboostOptions::TCatBoostOptions& catBoostOptions) {
        const bool needFeatureCombinations = (catBoostOptions.CatFeatureParams->MaxTensorComplexity > 1) && (catBoostOptions.BoostingOptions->DataPartitionType == EDataPartitionType::FeatureParallel);

        if (needFeatureCombinations) {
            const bool storeCatFeaturesInPinnedMemory = catBoostOptions.DataProcessingOptions->GpuCatFeaturesStorage ==
                                                        EGpuCatFeaturesStorage::CpuPinnedMemory;
            if (storeCatFeaturesInPinnedMemory) {
                ui32 devCount = NCudaLib::GetEnabledDevices(catBoostOptions.SystemOptions->Devices,
                                                            NCudaLib::GetDevicesProvider().GetDeviceCount())
                                    .size();
                ui32 cpuFeaturesSize = 104857600 + 1.05 * EstimatePinnedMemorySizeInBytesPerDevice(learn, test,
                                                                                                   featuresManager,
                                                                                                   devCount);
                ui64 currentSize = ParseMemorySizeDescription(catBoostOptions.SystemOptions->PinnedMemorySize.Get());
                if (currentSize < cpuFeaturesSize) {
                    catBoostOptions.SystemOptions->PinnedMemorySize = ToString(cpuFeaturesSize);
                }
            }
        }
    }

    inline void UpdateDataPartitionType(const TBinarizedFeaturesManager& featuresManager,
                                        NCatboostOptions::TCatBoostOptions& catBoostOptions) {
        if (catBoostOptions.CatFeatureParams->MaxTensorComplexity > 1 && featuresManager.GetCatFeatureIds().size()) {
            return;
        } else {
            if (catBoostOptions.BoostingOptions->BoostingType == EBoostingType::Plain) {
                if (catBoostOptions.BoostingOptions->DataPartitionType.NotSet()) {
                    catBoostOptions.BoostingOptions->DataPartitionType = EDataPartitionType::DocParallel;
                }
            }
        }
    }

    inline bool HasPermutationFeatures(const TBinarizedFeaturesManager& featuresManager) {
        for (auto catFeature : featuresManager.GetCatFeatureIds()) {
            if (featuresManager.UseForCtr(catFeature) || featuresManager.UseForTreeCtr(catFeature)) {
                return true;
            }
        }
        for (auto estimatedFeatureId : featuresManager.GetEstimatedFeatureIds()) {
            if (featuresManager.GetEstimatedFeature(estimatedFeatureId).EstimatorId.IsOnline) {
                return true;
            }
        }
        return false;
    }

    inline void UpdateGpuSpecificDefaults(NCatboostOptions::TCatBoostOptions& options,
                                          TBinarizedFeaturesManager& featuresManager) {
        //don't make several permutations in matrixnet-like mode if we don't have ctrs
        if (!HasPermutationFeatures(featuresManager) && options.BoostingOptions->BoostingType == EBoostingType::Plain) {
            if (options.BoostingOptions->PermutationCount > 1) {
                CATBOOST_DEBUG_LOG << "No catFeatures for ctrs found and don't look ahead is disabled. Fallback to one permutation" << Endl;
            }
            options.BoostingOptions->PermutationCount = 1;
        } else {
            if (options.BoostingOptions->PermutationCount > 1) {
                if (options.ObliviousTreeOptions->LeavesEstimationMethod.IsDefault() &&
                    options.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Simple)
                {
                    options.ObliviousTreeOptions->LeavesEstimationMethod.SetDefault(ELeavesEstimation::Newton);
                }
            }
        }

        NCatboostOptions::TOption<ui32>& blockSizeOption = options.BoostingOptions->PermutationBlockSize;
        if (!blockSizeOption.IsSet() || blockSizeOption == 0u) {
            blockSizeOption.Set(64);
        }
    }

    static inline bool NeedPriorEstimation(const TVector<NCatboostOptions::TCtrDescription>& descriptions) {
        for (const auto& description : descriptions) {
            if (description.PriorEstimation != EPriorEstimation::No) {
                return true;
            }
        }
        return false;
    }

    static inline const NCB::TGpuInputTargets* GetGpuInputTargets(const NCB::TTrainingDataProvider& dataProvider) {
        const auto* gpuObjects = dynamic_cast<const NCB::TGpuInputQuantizedObjectsDataProvider*>(dataProvider.ObjectsData.Get());
        if (!gpuObjects) {
            return nullptr;
        }
        const auto& gpuTargets = gpuObjects->GetGpuTargets();
        return gpuTargets.TargetCount ? &gpuTargets : nullptr;
    }

    static inline cudaStream_t GetCudaStreamFromCudaArrayInterface(ui64 stream) {
        if (stream == 0) {
            return 0;
        }
        if (stream == 1) {
            return cudaStreamPerThread;
        }
        return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
    }

    static inline EBorderSelectionType NormalizeBorderSelectionTypeForGpuInput(EBorderSelectionType borderSelectionType) {
        switch (borderSelectionType) {
            case EBorderSelectionType::GreedyLogSum:
            case EBorderSelectionType::GreedyMinEntropy:
            case EBorderSelectionType::Median:
                return EBorderSelectionType::Median;
            case EBorderSelectionType::Uniform:
                return EBorderSelectionType::Uniform;
            default:
                return EBorderSelectionType::Median;
        }
    }

    static TVector<float> ComputeFloatBordersFromGpuTarget(
        const NCB::TGpuInputColumnDesc& targetColumn,
        ui32 objectCount,
        const NCatboostOptions::TBinarizationOptions& binarizationOptions
    ) {
        CB_ENSURE(targetColumn.Data != 0, "GPU target pointer is null");
        CB_ENSURE(targetColumn.DeviceId >= 0, "Invalid GPU target device id");
        CB_ENSURE(targetColumn.StrideBytes > 0, "Invalid GPU target stride");

        const ui32 borderCount = binarizationOptions.BorderCount.Get();
        if (borderCount == 0 || objectCount == 0) {
            return {};
        }
        CB_ENSURE(borderCount <= 255, "BorderCount is too large for GPU target binarization");

        const EBorderSelectionType borderSelectionType = NormalizeBorderSelectionTypeForGpuInput(
            binarizationOptions.BorderSelectionType.Get()
        );

        CUDA_SAFE_CALL(cudaSetDevice(targetColumn.DeviceId));
        const cudaStream_t stream = GetCudaStreamFromCudaArrayInterface(targetColumn.Stream);

        float* values = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&values, static_cast<size_t>(objectCount) * sizeof(float)));
        Y_DEFER { cudaFree(values); };

        const void* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(targetColumn.Data));
        NKernel::CopyStridedGpuInputToFloat(
            src,
            targetColumn.StrideBytes,
            objectCount,
            static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(targetColumn.DType)),
            values,
            stream
        );
        CUDA_SAFE_CALL(cudaGetLastError());

        if (borderSelectionType == EBorderSelectionType::Uniform) {
            float minValue = 0.0f;
            float maxValue = 0.0f;
            NKernel::ComputeMinMaxToHost(values, objectCount, &minValue, &maxValue, stream);
            if (!std::isfinite(minValue) || !std::isfinite(maxValue) || (minValue == maxValue)) {
                return {};
            }

            TVector<float> borders;
            borders.yresize(borderCount);
            for (ui32 i = 0; i < borderCount; ++i) {
                const double v = static_cast<double>(minValue)
                    + (static_cast<double>(i) + 1.0)
                        * (static_cast<double>(maxValue) - static_cast<double>(minValue))
                        / (static_cast<double>(borderCount) + 1.0);
                borders[i] = static_cast<float>(v);
            }
            SortUnique(borders);
            return borders;
        }

        float* bordersDev = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&bordersDev, static_cast<size_t>(borderCount + 1) * sizeof(float)));
        Y_DEFER { cudaFree(bordersDev); };

        NKernel::FastGpuBorders(values, objectCount, bordersDev, borderCount, stream);
        CUDA_SAFE_CALL(cudaGetLastError());

        TVector<float> borders;
        borders.yresize(borderCount);
        NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(
            /*dst*/ borders.data(),
            /*src*/ bordersDev + 1,
            static_cast<ui64>(borderCount) * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            borders.data(),
            bordersDev + 1,
            static_cast<size_t>(borderCount) * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        ));
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

        EraseIf(borders, [] (float v) {
            return !std::isfinite(v);
        });
        SortUnique(borders);
        return borders;
    }

    static inline void EstimatePriors(const NCB::TTrainingDataProvider& dataProvider,
                                      TBinarizedFeaturesManager& featureManager,
                                      NCatboostOptions::TCatFeatureParams& options,
                                      NPar::ILocalExecutor* localExecutor) {
        CB_ENSURE(&(featureManager.GetCatFeatureOptions()) == &options, "Error: for consistent catFeature options should be equal to one in feature manager");

        bool needAnyPriorEstimation = NeedPriorEstimation(options.SimpleCtrs);
        if (options.PerFeatureCtrs.IsSet()) {
            for (const auto& [/*featureId*/_, ctrs] : options.PerFeatureCtrs.Get()) {
                if (NeedPriorEstimation(ctrs)) {
                    needAnyPriorEstimation = true;
                    break;
                }
            }
        }

        if (!needAnyPriorEstimation) {
            return;
        }

        bool needSimpleCtrsPriorEstimation = NeedPriorEstimation(options.SimpleCtrs);
        const auto& borders = featureManager.GetTargetBorders();
        if (borders.size() > 1) {
            return;
        }

        CB_ENSURE(
            dataProvider.TargetData->GetTarget().Defined(),
            "CTR prior estimation is not supported with GPU-resident targets; set prior_estimation=No"
        );

        const auto& featuresLayout = *dataProvider.MetaInfo.FeaturesLayout;

        auto binarizedTarget = NCB::BinarizeLine<ui8>((*dataProvider.TargetData->GetTarget())[0], ENanMode::Forbidden, borders); // espetrov: fix for multi-target + ctr
        TAdaptiveLock lock;

        //TODO(noxoomo): locks here are ugly and error prone
        NPar::ParallelFor(*localExecutor, 0, (int)featuresLayout.GetCatFeatureCount(), [&](int catFeatureIdx) {
            if (!featuresLayout.GetInternalFeatureMetaInfo((ui32)catFeatureIdx, EFeatureType::Categorical).IsAvailable) {
                return;
            }
            const auto& catFeatureValues = **(dataProvider.ObjectsData->GetCatFeature(catFeatureIdx));

            bool hasPerFeatureCtr = false;

            auto catFeatureFlatIdx = featuresLayout.GetExternalFeatureIdx(catFeatureIdx, EFeatureType::Categorical);

            with_lock (lock) {
                if (needSimpleCtrsPriorEstimation && !options.PerFeatureCtrs->contains(catFeatureFlatIdx)) {
                    options.PerFeatureCtrs.Get()[catFeatureFlatIdx] = options.SimpleCtrs;
                }
                hasPerFeatureCtr = options.PerFeatureCtrs->contains(catFeatureFlatIdx);
            }

            if (hasPerFeatureCtr) {
                TVector<NCatboostOptions::TCtrDescription> currentFeatureDescription;
                with_lock (lock) {
                    currentFeatureDescription = options.PerFeatureCtrs->at(catFeatureFlatIdx);
                }
                if (!NeedPriorEstimation(currentFeatureDescription)) {
                    return;
                }

                for (ui32 i = 0; i < currentFeatureDescription.size(); ++i) {
                    if (currentFeatureDescription[i].Type == ECtrType::Borders && options.TargetBinarization->BorderCount == 1u) {
                        ui32 uniqueValues = dataProvider.ObjectsData->GetQuantizedFeaturesInfo()->GetUniqueValuesCounts(TCatFeatureIdx((ui32)catFeatureIdx)).OnAll;

                        TBetaPriorEstimator::TBetaPrior prior = TBetaPriorEstimator::EstimateBetaPrior(
                            binarizedTarget.data(),
                            catFeatureValues.GetBlockIterator(),
                            catFeatureValues.GetSize(),
                            uniqueValues
                        );

                        CATBOOST_INFO_LOG << "Estimate borders-ctr prior for feature #" << catFeatureFlatIdx << ": " << prior.Alpha << " / " << prior.Beta << Endl;
                        currentFeatureDescription[i].Priors = {{(float)prior.Alpha, (float)(prior.Alpha + prior.Beta)}};
                    } else {
                        CB_ENSURE(currentFeatureDescription[i].PriorEstimation == EPriorEstimation::No, "Error: auto prior estimation is not available for ctr type " << currentFeatureDescription[i].Type);
                    }
                }
                with_lock (lock) {
                    options.PerFeatureCtrs.Get()[catFeatureFlatIdx] = currentFeatureDescription;
                }
            }
        });
    }

    static void SetDataDependentDefaultsForGpu(const NCB::TTrainingDataProvider& dataProvider,
                                               const NCB::TTrainingDataProvider* testProvider,
                                               NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                               TBinarizedFeaturesManager& featuresManager,
                                               NPar::ILocalExecutor* localExecutor) {
        UpdateGpuSpecificDefaults(catBoostOptions, featuresManager);
        EstimatePriors(dataProvider, featuresManager, catBoostOptions.CatFeatureParams, localExecutor);
        UpdateDataPartitionType(featuresManager, catBoostOptions);
        UpdatePinnedMemorySizeOption(dataProvider, testProvider, featuresManager, catBoostOptions);
    }

    static void ConfigureCudaProfiler(bool isProfile, NCudaLib::TCudaProfiler* profiler) {
        if (isProfile) {
            profiler->SetDefaultProfileMode(NCudaLib::EProfileMode::ImplicitLabelSync);
        } else {
            profiler->SetDefaultProfileMode(NCudaLib::EProfileMode::NoProfile);
        }
    }

    TGpuTrainResult TrainModelImpl(const TTrainModelInternalOptions& internalOptions,
                                                                const NCatboostOptions::TCatBoostOptions& trainCatBoostOptions,
                                                                const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                const TTrainingDataProvider& dataProvider,
                                                                const TTrainingDataProvider* testProvider,
                                                                const TFeatureEstimators& featureEstimators,
                                                                TBinarizedFeaturesManager& featuresManager,
                                                                ui32 approxDimension,
                                                                ITrainingCallbacks* trainingCallbacks,
                                                                NPar::ILocalExecutor* localExecutor,
                                                                TVector<TVector<double>>* testMultiApprox, // [dim][objectIdx]
                                                                TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
                                                                const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                                                                const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor
                                                                ) {
        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();
        ConfigureCudaProfiler(trainCatBoostOptions.IsProfile, &profiler);

        TGpuAwareRandom random(trainCatBoostOptions.RandomSeed);

        TGpuTrainResult model;

        const auto optimizationImplementation = GetTrainerFactoryKey(trainCatBoostOptions);

        if (TGpuTrainerFactory::Has(optimizationImplementation)) {
            THolder<IGpuTrainer> trainer(TGpuTrainerFactory::Construct(optimizationImplementation));
            model = trainer->TrainModel(featuresManager,
                                        internalOptions,
                                        trainCatBoostOptions,
                                        outputOptions,
                                        dataProvider,
                                        testProvider,
                                        featureEstimators,
                                        objectiveDescriptor,
                                        evalMetricDescriptor,
                                        random,
                                        approxDimension,
                                        trainingCallbacks,
                                        localExecutor,
                                        testMultiApprox,
                                        metricsAndTimeHistory);
        } else {
            ythrow TCatBoostException() << "Error: optimization scheme is not supported for GPU learning " << optimizationImplementation;
        }
        return model;
    }

    void ModelBasedEvalImpl(const NCatboostOptions::TCatBoostOptions& trainCatBoostOptions,
                            const NCatboostOptions::TOutputFilesOptions& outputOptions,
                            const TTrainingDataProvider& dataProvider,
                            const TTrainingDataProvider& testProvider,
                            TBinarizedFeaturesManager& featuresManager,
                            ui32 approxDimension,
                            NPar::ILocalExecutor* localExecutor) {
        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

        ConfigureCudaProfiler(trainCatBoostOptions.IsProfile, &profiler);

        const auto optimizationImplementation = GetTrainerFactoryKey(trainCatBoostOptions);
        CB_ENSURE(TGpuTrainerFactory::Has(optimizationImplementation),
            "Error: optimization scheme is not supported for GPU learning " << optimizationImplementation);
        THolder<IGpuTrainer> trainer(TGpuTrainerFactory::Construct(optimizationImplementation));
        TGpuAwareRandom random(trainCatBoostOptions.RandomSeed);
        trainer->ModelBasedEval(featuresManager,
            trainCatBoostOptions,
            outputOptions,
            dataProvider,
            testProvider,
            random,
            approxDimension,
            localExecutor);
    }

    class TGPUModelTrainer: public IModelTrainer {
    public:
        void TrainModel(
            const TTrainModelInternalOptions& internalOptions,
            const NCatboostOptions::TCatBoostOptions& catboostOptions,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
            const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
            TTrainingDataProviders trainingData,
            TMaybe<NCB::TPrecomputedOnlineCtrData> precomputedSingleOnlineCtrDataForSingleFold,
            const TLabelConverter& labelConverter,
            ITrainingCallbacks* trainingCallbacks,
            ICustomCallbacks* /*customCallbacks*/,
            TMaybe<TFullModel*> initModel,
            THolder<TLearnProgress> initLearnProgress,
            NCB::TDataProviders initModelApplyCompatiblePools,
            NPar::ILocalExecutor* localExecutor,
            const TMaybe<TRestorableFastRng64*> rand,
            TFullModel* dstModel,
            const TVector<TEvalResult*>& evalResultPtrs,
            TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
            THolder<TLearnProgress>* dstLearnProgress) const override {

            Y_UNUSED(rand);
            CB_ENSURE(trainingData.Test.size() <= 1, "Multiple eval sets not supported for GPU");
            CB_ENSURE(!precomputedSingleOnlineCtrDataForSingleFold,
                      "Precomputed online CTR data for GPU is not yet supported");
            CB_ENSURE(
                evalResultPtrs.empty() || (evalResultPtrs.size() == trainingData.Test.size()),
                "Need test dataset to evaluate resulting model");
            CB_ENSURE(!initModel && !initLearnProgress, "Training continuation for GPU is not yet supported");
            Y_UNUSED(initModelApplyCompatiblePools);
            CB_ENSURE_INTERNAL(!dstLearnProgress, "Returning learn progress for GPU is not yet supported");

            NCatboostOptions::TCatBoostOptions updatedCatboostOptions(catboostOptions);

            bool saveFinalCtrsInModel = !internalOptions.CalcMetricsOnly &&
                (outputOptions.GetFinalCtrComputationMode() == EFinalCtrComputationMode::Default) &&
                (trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo()
                    ->CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn()
                  > updatedCatboostOptions.CatFeatureParams.Get().OneHotMaxSize.Get());

            auto quantizedFeaturesInfo = trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo();
            TVector<TExclusiveFeaturesBundle> exclusiveBundlesCopy;
            const auto lossFunction = catboostOptions.LossFunctionDescription->LossFunction;
            // TODO(kirillovs): check and enable on pairwise losses
            if (!IsGpuPlainDocParallelOnlyMode(lossFunction)) {
                exclusiveBundlesCopy.assign(
                    trainingData.Learn->ObjectsData->GetExclusiveFeatureBundlesMetaData().begin(),
                    trainingData.Learn->ObjectsData->GetExclusiveFeatureBundlesMetaData().end()
                );
            }
            ui32 objectsCount = trainingData.Learn->GetObjectCount();
            if (!trainingData.Test.empty()) {
                objectsCount += trainingData.Test[0]->GetObjectCount();
            }
            TBinarizedFeaturesManager featuresManager(updatedCatboostOptions.CatFeatureParams,
                                                      trainingData.FeatureEstimators,
                                                      *trainingData.Learn->MetaInfo.FeaturesLayout,
                                                      exclusiveBundlesCopy,
                                                      quantizedFeaturesInfo,
                                                      objectsCount,
                                                      /*enableShuffling*/internalOptions.HaveLearnFeatureInMemory);

            SetDataDependentDefaultsForGpu(
                *trainingData.Learn,
                !trainingData.Test.empty() ? trainingData.Test[0].Get() : nullptr,
                updatedCatboostOptions,
                featuresManager,
                localExecutor);

            const TString& trainingOptionsFileName = outputOptions.CreateTrainingOptionsFullPath();
            if (!trainingOptionsFileName.empty()) {
                TOFStream trainingOptionsFile(trainingOptionsFileName);
                trainingOptionsFile.Write(NJson::PrettifyJson(ToString(updatedCatboostOptions)));
            }

            if (const auto* gpuTargets = GetGpuInputTargets(*trainingData.Learn)) {
                featuresManager.SetTargetBorders(
                    ComputeFloatBordersFromGpuTarget(
                        gpuTargets->Targets[0],
                        trainingData.Learn->GetObjectCount(),
                        featuresManager.GetTargetBinarizationDescription()
                    )
                );
            } else {
                NCB::TOnCpuGridBuilderFactory gridBuilderFactory;
                featuresManager.SetTargetBorders(
                    NCB::TBordersBuilder(
                        gridBuilderFactory,
                        (*trainingData.Learn->TargetData->GetTarget())[0]
                    )(featuresManager.GetTargetBinarizationDescription())
                ); // esp: fix for multi-target
            }

            TSetLogging inThisScope(updatedCatboostOptions.LoggingLevel);

            const ui32 targetDimension = [&] () -> ui32 {
                if (const auto* gpuTargets = GetGpuInputTargets(*trainingData.Learn)) {
                    return gpuTargets->TargetCount;
                }
                return trainingData.Learn->TargetData->GetTargetDimension();
            }();
            ui32 approxDimension = GetApproxDimension(
                updatedCatboostOptions,
                labelConverter,
                targetDimension
            );

            TVector<TVector<double>> rawValues(approxDimension);

            CheckMetrics(updatedCatboostOptions.MetricOptions);

            TGpuTrainResult gpuFormatModel = TrainModelImpl(
                internalOptions,
                updatedCatboostOptions,
                outputOptions,
                *trainingData.Learn,
                !trainingData.Test.empty() ? trainingData.Test[0].Get() : nullptr,
                *trainingData.FeatureEstimators,
                featuresManager,
                approxDimension,
                trainingCallbacks,
                localExecutor,
                &rawValues,
                metricsAndTimeHistory,
                objectiveDescriptor,
                evalMetricDescriptor
            );

            if (evalResultPtrs.size()) {
                evalResultPtrs[0]->SetRawValuesByMove(rawValues);
            }

            if (internalOptions.CalcMetricsOnly) {
                return;
            }

            TPerfectHashedToHashedCatValuesMap perfectHashedToHashedCatValuesMap = quantizedFeaturesInfo->CalcPerfectHashedToHashedCatValuesMap(localExecutor);
            if (outputOptions.AllowWriteFiles()) {
                TString tmpDir;
                NCB::NPrivate::CreateTrainDirWithTmpDirIfNotExist(outputOptions.GetTrainDir(), &tmpDir);
                quantizedFeaturesInfo->UnloadCatFeaturePerfectHashFromRam(tmpDir);
            }

            TClassificationTargetHelper classificationTargetHelper(labelConverter,
                                                                   updatedCatboostOptions.DataProcessingOptions);

            TMaybe<TFullModel> fullModel;
            TFullModel* modelPtr = nullptr;
            if (dstModel) {
                modelPtr = dstModel;
            } else {
                fullModel.ConstructInPlace();
                modelPtr = &*fullModel;
            }

            THashMap<TFeatureCombination, TProjection> featureCombinationToProjection;
            if (std::holds_alternative<THolder<TAdditiveModel<TObliviousTreeModel>>>(gpuFormatModel)) {
                auto& modelHolderRef = std::get<THolder<TAdditiveModel<TObliviousTreeModel>>>(gpuFormatModel);
                *modelPtr = ConvertToCoreModel(featuresManager,
                                               quantizedFeaturesInfo,
                                               perfectHashedToHashedCatValuesMap,
                                               classificationTargetHelper,
                                               *modelHolderRef,
                                               &featureCombinationToProjection);

                modelHolderRef.Destroy();
            } else {
                auto& modelHolderRef = std::get<THolder<TAdditiveModel<TNonSymmetricTree>>>(gpuFormatModel);
                *modelPtr = ConvertToCoreModel(featuresManager,
                                               quantizedFeaturesInfo,
                                               perfectHashedToHashedCatValuesMap,
                                               classificationTargetHelper,
                                               *modelHolderRef,
                                               &featureCombinationToProjection);

                modelHolderRef.Destroy();
            }

            auto targetClassifiers = CreateTargetClassifiers(featuresManager);

            EFinalFeatureCalcersComputationMode featureCalcerComputationMode = outputOptions.GetFinalFeatureCalcerComputationMode();
            if (modelPtr->ModelTrees->GetTextFeatures().empty() &&
                modelPtr->ModelTrees->GetEstimatedFeatures().empty()
            ) {
                featureCalcerComputationMode = EFinalFeatureCalcersComputationMode::Skip;
            }

            TCoreModelToFullModelConverter coreModelToFullModelConverter(
                updatedCatboostOptions,
                outputOptions,
                classificationTargetHelper,
                /*ctrLeafCountLimit*/ Max<ui64>(),
                /*storeAllSimpleCtrs*/ false,
                saveFinalCtrsInModel ? EFinalCtrComputationMode::Default : EFinalCtrComputationMode::Skip,
                featureCalcerComputationMode);

            coreModelToFullModelConverter.WithBinarizedDataComputedFrom(
                                             trainingData,
                                             std::move(featureCombinationToProjection),
                                             targetClassifiers)
                .WithPerfectHashedToHashedCatValuesMap(
                    &perfectHashedToHashedCatValuesMap)
                .WithCoreModelFrom(
                    modelPtr)
                .WithObjectsDataFrom(trainingData.Learn->ObjectsData)
                .WithFeatureEstimators(trainingData.FeatureEstimators)
                .WithMetrics(*metricsAndTimeHistory);

            if (dstModel) {
                coreModelToFullModelConverter.Do(true, dstModel, localExecutor, &targetClassifiers);
            } else {
                coreModelToFullModelConverter.Do(
                    outputOptions.CreateResultModelFullPath(),
                    outputOptions.GetModelFormats(),
                    outputOptions.AddFileFormatExtension(),
                    localExecutor,
                    &targetClassifiers);
            }
        }

        void ModelBasedEval(
            const NCatboostOptions::TCatBoostOptions& catboostOptions,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            TTrainingDataProviders trainingData,
            const TLabelConverter& labelConverter,
            NPar::ILocalExecutor* localExecutor) const override {

            CB_ENSURE(trainingData.Test.size() == 1, "Model based evaluation requires exactly one eval set on GPU");
            CB_ENSURE(!IsMultiTargetObjective(catboostOptions.LossFunctionDescription->LossFunction),
                      "Catboost does not support multitarget on GPU yet");

            NCatboostOptions::TCatBoostOptions updatedCatboostOptions(catboostOptions);

            auto quantizedFeaturesInfo = trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo();

            TFeatureEstimatorsPtr estimators;
            TVector<TExclusiveFeaturesBundle> exclusiveBundlesCopy;
            // TODO(kirillovs): check and enable in modelbased eval
            /*exclusiveBundlesCopy.assign(
                trainingData.Learn->ObjectsData->GetExclusiveFeatureBundlesMetaData().begin(),
                trainingData.Learn->ObjectsData->GetExclusiveFeatureBundlesMetaData().end()
            );*/
            ui32 objectsCount = trainingData.Learn->GetObjectCount() + trainingData.Test[0]->GetObjectCount();
            TBinarizedFeaturesManager featuresManager(updatedCatboostOptions.CatFeatureParams,
                                                      estimators,
                                                      *trainingData.Learn->MetaInfo.FeaturesLayout,
                                                      exclusiveBundlesCopy,
                                                      quantizedFeaturesInfo,
                                                      objectsCount);

            SetDataDependentDefaultsForGpu(
                *trainingData.Learn,
                trainingData.Test[0].Get(),
                updatedCatboostOptions,
                featuresManager,
                localExecutor);

            if (const auto* gpuTargets = GetGpuInputTargets(*trainingData.Learn)) {
                featuresManager.SetTargetBorders(
                    ComputeFloatBordersFromGpuTarget(
                        gpuTargets->Targets[0],
                        trainingData.Learn->GetObjectCount(),
                        featuresManager.GetTargetBinarizationDescription()
                    )
                );
            } else {
                NCB::TOnCpuGridBuilderFactory gridBuilderFactory;
                featuresManager.SetTargetBorders(
                    NCB::TBordersBuilder(
                        gridBuilderFactory,
                        *trainingData.Learn->TargetData->GetOneDimensionalTarget()
                    )(featuresManager.GetTargetBinarizationDescription())
                );
            }

            TSetLogging inThisScope(updatedCatboostOptions.LoggingLevel);
            auto deviceRequestConfig = NCudaLib::CreateDeviceRequestConfig(updatedCatboostOptions);
            auto stopCudaManagerGuard = StartCudaManager(deviceRequestConfig,
                                                         updatedCatboostOptions.LoggingLevel);

            const ui32 targetDimension = [&] () -> ui32 {
                if (const auto* gpuTargets = GetGpuInputTargets(*trainingData.Learn)) {
                    return gpuTargets->TargetCount;
                }
                return trainingData.Learn->TargetData->GetTargetDimension();
            }();
            ui32 approxDimension = GetApproxDimension(
                updatedCatboostOptions,
                labelConverter,
                targetDimension
            );

            CheckMetrics(updatedCatboostOptions.MetricOptions);

            ModelBasedEvalImpl(
                updatedCatboostOptions,
                outputOptions,
                *trainingData.Learn,
                *trainingData.Test[0].Get(),
                featuresManager,
                approxDimension,
                localExecutor);
        }
    };

}

TTrainerFactory::TRegistrator<NCatboostCuda::TGPUModelTrainer> GPURegistrator(ETaskType::GPU);

template <>
inline TString ToString<NCatboostCuda::TGpuTrainerFactoryKey>(const NCatboostCuda::TGpuTrainerFactoryKey& key) {
    return TStringBuilder() << "Loss=" << key.Loss << ";OptimizationScheme=" << key.GrowPolicy;
}

template <>
void Out<NCatboostCuda::TGpuTrainerFactoryKey>(IOutputStream& o, const NCatboostCuda::TGpuTrainerFactoryKey& key) {
    o.Write(ToString(key));
}
