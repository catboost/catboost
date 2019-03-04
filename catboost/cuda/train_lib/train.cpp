#include "train.h"

#include <catboost/cuda/cpu_compatibility_helpers/model_converter.h>
#include <catboost/cuda/ctrs/prior_estimator.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_lib/devices_provider.h>
#include <catboost/cuda/gpu_data/pinned_memory_estimation.h>

#include <catboost/libs/algo/custom_objective_descriptor.h>
#include <catboost/libs/algo/full_model_saver.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/algo/online_ctr.h>
#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/fstr/output_fstr.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/options/defaults_helper.h>
#include <catboost/libs/options/metric_options.h>
#include <catboost/libs/options/system_options.h>
#include <catboost/libs/quantization/grid_creator.h>
#include <catboost/libs/quantization/utils.h>
#include <catboost/libs/train_lib/approx_dimension.h>
#include <catboost/libs/train_lib/preprocess.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/json/json_value.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/folder/path.h>
#include <util/generic/scope.h>
#include <util/system/compiler.h>
#include <util/system/guard.h>
#include <util/system/info.h>
#include <util/system/spinlock.h>
#include <util/system/yassert.h>
#include <catboost/cuda/models/compact_model.h>

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

    inline bool HasCtrs(const TBinarizedFeaturesManager& featuresManager) {
        for (auto catFeature : featuresManager.GetCatFeatureIds()) {
            if (featuresManager.UseForCtr(catFeature) || featuresManager.UseForTreeCtr(catFeature)) {
                return true;
            }
        }
        return false;
    }

    inline void UpdateGpuSpecificDefaults(NCatboostOptions::TCatBoostOptions& options,
                                          TBinarizedFeaturesManager& featuresManager) {
        //don't make several permutations in matrixnet-like mode if we don't have ctrs
        if (!HasCtrs(featuresManager) && options.BoostingOptions->BoostingType == EBoostingType::Plain) {
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

    inline NCudaLib::TDeviceRequestConfig CreateDeviceRequestConfig(const NCatboostOptions::TCatBoostOptions& options) {
        NCudaLib::TDeviceRequestConfig config;
        const auto& systemOptions = options.SystemOptions.Get();
        config.DeviceConfig = systemOptions.Devices;
        config.PinnedMemorySize = ParseMemorySizeDescription(systemOptions.PinnedMemorySize.Get());
        config.GpuMemoryPartByWorker = systemOptions.GpuRamPart;
        return config;
    }

    static inline bool NeedPriorEstimation(const TVector<NCatboostOptions::TCtrDescription>& descriptions) {
        for (const auto& description : descriptions) {
            if (description.PriorEstimation != EPriorEstimation::No) {
                return true;
            }
        }
        return false;
    }

    static inline void EstimatePriors(const NCB::TTrainingDataProvider& dataProvider,
                                      TBinarizedFeaturesManager& featureManager,
                                      NCatboostOptions::TCatFeatureParams& options,
                                      NPar::TLocalExecutor* localExecutor) {
        CB_ENSURE(&(featureManager.GetCatFeatureOptions()) == &options, "Error: for consistent catFeature options should be equal to one in feature manager");

        bool needSimpleCtrsPriorEstimation = NeedPriorEstimation(options.SimpleCtrs);
        const auto& borders = featureManager.GetTargetBorders();
        if (borders.size() > 1) {
            return;
        }

        const auto& featuresLayout = *dataProvider.MetaInfo.FeaturesLayout;

        auto binarizedTarget = NCB::BinarizeLine<ui8>(GetTarget(dataProvider.TargetData), ENanMode::Forbidden, borders);

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
                auto values = catFeatureValues.ExtractValues(localExecutor);

                for (ui32 i = 0; i < currentFeatureDescription.size(); ++i) {
                    if (currentFeatureDescription[i].Type == ECtrType::Borders && options.TargetBinarization->BorderCount == 1u) {
                        ui32 uniqueValues = dataProvider.ObjectsData->GetQuantizedFeaturesInfo()->GetUniqueValuesCounts(TCatFeatureIdx((ui32)catFeatureIdx)).OnAll;

                        TBetaPriorEstimator::TBetaPrior prior = TBetaPriorEstimator::EstimateBetaPrior(binarizedTarget.data(),
                                                                                                       (*values).data(), (*values).size(), uniqueValues);

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
                                               NCatboostOptions::TOutputFilesOptions& outputOptions,
                                               TBinarizedFeaturesManager& featuresManager,
                                               NPar::TLocalExecutor* localExecutor) {
        bool hasTestConstTarget = true;
        bool hasTestPairs = false;
        ui32 testPoolSize = 0;
        if (testProvider) {
            hasTestConstTarget = IsConst(GetTarget(testProvider->TargetData));
            hasTestPairs = testProvider->TargetData.contains(TTargetDataSpecification(ETargetType::GroupPairwiseRanking));
            testPoolSize = testProvider->GetObjectCount();
        }

        SetDataDependentDefaults(dataProvider.GetObjectCount(),
                                 dataProvider.MetaInfo.HasTarget,
                                 dataProvider.ObjectsData->GetQuantizedFeaturesInfo()
                                     ->CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn(),
                                 testPoolSize,
                                 hasTestConstTarget,
                                 hasTestPairs,
                                 &outputOptions.UseBestModel,
                                 &catBoostOptions);

        UpdateGpuSpecificDefaults(catBoostOptions, featuresManager);
        EstimatePriors(dataProvider, featuresManager, catBoostOptions.CatFeatureParams, localExecutor);
        UpdateDataPartitionType(featuresManager, catBoostOptions);
        UpdatePinnedMemorySizeOption(dataProvider, testProvider, featuresManager, catBoostOptions);
    }

    static void WarnIfUseToSymmetricTrees(const NCatboostOptions::TCatBoostOptions& trainCatBoostOptions) {
        if (trainCatBoostOptions.ObliviousTreeOptions->GrowingPolicy == EGrowingPolicy::Lossguide) {
            if (trainCatBoostOptions.ObliviousTreeOptions->MaxLeavesCount > 64) {
                CATBOOST_WARNING_LOG << "Warning: CatBoost will need to convert non symmetric tree to symmetric one currently. With big number of leaves model conversion could fail or model size could be very big" << Endl;
            }
        }
        if (trainCatBoostOptions.ObliviousTreeOptions->GrowingPolicy == EGrowingPolicy::Levelwise) {
            if (trainCatBoostOptions.ObliviousTreeOptions->MaxDepth > 10) {
                CATBOOST_WARNING_LOG << "Warning: CatBoost will need to convert non symmetric tree to symmetric one currently. With deep trees model conversion could fail or model size could be very big" << Endl;
            }
        }
    }

    static void ConfigureCudaProfiler(bool isProfile, NCudaLib::TCudaProfiler* profiler) {
        if (isProfile) {
            profiler->SetDefaultProfileMode(NCudaLib::EProfileMode::ImplicitLabelSync);
        } else {
            profiler->SetDefaultProfileMode(NCudaLib::EProfileMode::NoProfile);
        }
    }

    THolder<TAdditiveModel<TObliviousTreeModel>> TrainModelImpl(const TTrainModelInternalOptions& internalOptions,
                                                                const NCatboostOptions::TCatBoostOptions& trainCatBoostOptions,
                                                                const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                const TTrainingDataProvider& dataProvider,
                                                                const TTrainingDataProvider* testProvider,
                                                                TBinarizedFeaturesManager& featuresManager,
                                                                ui32 approxDimension,
                                                                const TMaybe<TOnEndIterationCallback>& onEndIterationCallback,
                                                                NPar::TLocalExecutor* localExecutor,
                                                                TVector<TVector<double>>* testMultiApprox, // [dim][objectIdx]
                                                                TMetricsAndTimeLeftHistory* metricsAndTimeHistory) {
        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();
        ConfigureCudaProfiler(trainCatBoostOptions.IsProfile, &profiler);

        TGpuAwareRandom random(trainCatBoostOptions.RandomSeed);

        THolder<TAdditiveModel<TObliviousTreeModel>> model;

        const auto optimizationImplementation = GetTrainerFactoryKey(trainCatBoostOptions);

        WarnIfUseToSymmetricTrees(trainCatBoostOptions);

        if (TGpuTrainerFactory::Has(optimizationImplementation)) {
            THolder<IGpuTrainer> trainer = TGpuTrainerFactory::Construct(optimizationImplementation);
            model = trainer->TrainModel(featuresManager,
                                        internalOptions,
                                        trainCatBoostOptions,
                                        outputOptions,
                                        dataProvider,
                                        testProvider,
                                        random,
                                        approxDimension,
                                        onEndIterationCallback,
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
                            NPar::TLocalExecutor* localExecutor) {
        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

        ConfigureCudaProfiler(trainCatBoostOptions.IsProfile, &profiler);

        WarnIfUseToSymmetricTrees(trainCatBoostOptions);

        const auto optimizationImplementation = GetTrainerFactoryKey(trainCatBoostOptions);
        CB_ENSURE(TGpuTrainerFactory::Has(optimizationImplementation),
            "Error: optimization scheme is not supported for GPU learning " << optimizationImplementation);
        THolder<IGpuTrainer> trainer = TGpuTrainerFactory::Construct(optimizationImplementation);
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

    inline void CreateDirIfNotExist(const TString& path) {
        TFsPath trainDirPath(path);
        try {
            if (!path.empty() && !trainDirPath.Exists()) {
                trainDirPath.MkDir();
            }
        } catch (...) {
            ythrow TCatBoostException() << "Can't create working dir: " << path;
        }
    }

    class TGPUModelTrainer: public IModelTrainer {
    public:
        void TrainModel(
            const TTrainModelInternalOptions& internalOptions,
            const NJson::TJsonValue& params,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
            const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
            const TMaybe<TOnEndIterationCallback>& onEndIterationCallback,
            TTrainingDataProviders trainingData,
            const TLabelConverter& labelConverter,
            NPar::TLocalExecutor* localExecutor,
            const TMaybe<TRestorableFastRng64*> rand,
            TFullModel* model,
            const TVector<TEvalResult*>& evalResultPtrs,
            TMetricsAndTimeLeftHistory* metricsAndTimeHistory) const override {
            Y_UNUSED(objectiveDescriptor);
            Y_UNUSED(evalMetricDescriptor);
            Y_UNUSED(rand);
            CB_ENSURE(trainingData.Test.size() <= 1, "Multiple eval sets not supported for GPU");
            Y_VERIFY(evalResultPtrs.size() == trainingData.Test.size());

            NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::GPU);
            catBoostOptions.Load(params);

            bool saveFinalCtrsInModel = !internalOptions.CalcMetricsOnly &&
                (outputOptions.GetFinalCtrComputationMode() == EFinalCtrComputationMode::Default) &&
                (trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo()
                    ->CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn()
                  > catBoostOptions.CatFeatureParams.Get().OneHotMaxSize.Get());

            TTrainingForCPUDataProviders trainingDataForFinalCtrCalculation;

            if (saveFinalCtrsInModel) {
                // do it at this stage to check before training
                trainingDataForFinalCtrCalculation = trainingData.Cast<TQuantizedForCPUObjectsDataProvider>();
            }

            auto quantizedFeaturesInfo = trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo();

            TBinarizedFeaturesManager featuresManager(catBoostOptions.CatFeatureParams,
                                                      quantizedFeaturesInfo);

            NCatboostOptions::TOutputFilesOptions updatedOutputOptions = outputOptions;

            SetDataDependentDefaultsForGpu(
                *trainingData.Learn,
                !trainingData.Test.empty() ? trainingData.Test[0].Get() : nullptr,
                catBoostOptions,
                updatedOutputOptions,
                featuresManager,
                localExecutor);

            NCB::TOnCpuGridBuilderFactory gridBuilderFactory;
            featuresManager.SetTargetBorders(
                NCB::TBordersBuilder(
                    gridBuilderFactory,
                    GetTarget(trainingData.Learn->TargetData))(featuresManager.GetTargetBinarizationDescription()));

            TSetLogging inThisScope(catBoostOptions.LoggingLevel);
            CreateDirIfNotExist(updatedOutputOptions.GetTrainDir());
            auto deviceRequestConfig = CreateDeviceRequestConfig(catBoostOptions);
            auto stopCudaManagerGuard = StartCudaManager(deviceRequestConfig,
                                                         catBoostOptions.LoggingLevel);

            ui32 approxDimension = GetApproxDimension(catBoostOptions, labelConverter);

            TVector<TVector<double>> rawValues(approxDimension);

            THolder<TAdditiveModel<TObliviousTreeModel>> gpuFormatModel = TrainModelImpl(
                internalOptions,
                catBoostOptions,
                updatedOutputOptions,
                *trainingData.Learn,
                !trainingData.Test.empty() ? trainingData.Test[0].Get() : nullptr,
                featuresManager,
                approxDimension,
                onEndIterationCallback,
                localExecutor,
                &rawValues,
                metricsAndTimeHistory);

            if (evalResultPtrs.size()) {
                evalResultPtrs[0]->SetRawValuesByMove(rawValues);
            }

            if (internalOptions.CalcMetricsOnly) {
                return;
            }

            TPerfectHashedToHashedCatValuesMap perfectHashedToHashedCatValuesMap = quantizedFeaturesInfo->CalcPerfectHashedToHashedCatValuesMap(localExecutor);

            TClassificationTargetHelper classificationTargetHelper(labelConverter,
                                                                   catBoostOptions.DataProcessingOptions);

            TMaybe<TFullModel> fullModel;
            TFullModel* modelPtr = nullptr;
            if (model) {
                modelPtr = model;
            } else {
                fullModel.ConstructInPlace();
                modelPtr = &*fullModel;
            }

            THashMap<TFeatureCombination, TProjection> featureCombinationToProjection;
            *modelPtr = ConvertToCoreModel(featuresManager,
                                           quantizedFeaturesInfo,
                                           perfectHashedToHashedCatValuesMap,
                                           classificationTargetHelper,
                                           *gpuFormatModel,
                                           &featureCombinationToProjection);

            gpuFormatModel.Destroy();

            auto targetClassifiers = CreateTargetClassifiers(featuresManager);

            TCoreModelToFullModelConverter coreModelToFullModelConverter(
                catBoostOptions,
                classificationTargetHelper,
                /*ctrLeafCountLimit*/ Max<ui64>(),
                /*storeAllSimpleCtrs*/ false,
                saveFinalCtrsInModel ? EFinalCtrComputationMode::Default : EFinalCtrComputationMode::Skip);

            coreModelToFullModelConverter.WithBinarizedDataComputedFrom(
                                             std::move(trainingDataForFinalCtrCalculation),
                                             std::move(featureCombinationToProjection),
                                             targetClassifiers)
                .WithPerfectHashedToHashedCatValuesMap(
                    &perfectHashedToHashedCatValuesMap)
                .WithCoreModelFrom(
                    modelPtr)
                .WithObjectsDataFrom(trainingData.Learn->ObjectsData);

            if (model) {
                coreModelToFullModelConverter.Do(true, model);
            } else {
                coreModelToFullModelConverter.Do(
                    updatedOutputOptions.CreateResultModelFullPath(),
                    updatedOutputOptions.GetModelFormats(),
                    updatedOutputOptions.AddFileFormatExtension());
            }
        }

        void ModelBasedEval(
            const NJson::TJsonValue& params,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            TTrainingDataProviders trainingData,
            const TLabelConverter& labelConverter,
            NPar::TLocalExecutor* localExecutor) const override {
            CB_ENSURE(trainingData.Test.size() == 1, "Model based evaluation requires exactly one eval set on GPU");

            NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::GPU);
            catBoostOptions.Load(params);

            auto quantizedFeaturesInfo = trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo();

            TBinarizedFeaturesManager featuresManager(catBoostOptions.CatFeatureParams,
                                                      quantizedFeaturesInfo);

            NCatboostOptions::TOutputFilesOptions updatedOutputOptions = outputOptions;

            SetDataDependentDefaultsForGpu(
                *trainingData.Learn,
                trainingData.Test[0].Get(),
                catBoostOptions,
                updatedOutputOptions,
                featuresManager,
                localExecutor);

            NCB::TOnCpuGridBuilderFactory gridBuilderFactory;
            featuresManager.SetTargetBorders(
                NCB::TBordersBuilder(
                    gridBuilderFactory,
                    GetTarget(trainingData.Learn->TargetData))(featuresManager.GetTargetBinarizationDescription()));

            TSetLogging inThisScope(catBoostOptions.LoggingLevel);
            CreateDirIfNotExist(updatedOutputOptions.GetTrainDir());
            auto deviceRequestConfig = CreateDeviceRequestConfig(catBoostOptions);
            auto stopCudaManagerGuard = StartCudaManager(deviceRequestConfig,
                                                         catBoostOptions.LoggingLevel);

            ui32 approxDimension = GetApproxDimension(catBoostOptions, labelConverter);

            ModelBasedEvalImpl(
                catBoostOptions,
                updatedOutputOptions,
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
    return TStringBuilder() << "Loss=" << key.Loss << ";OptimizationScheme=" << key.GrowingPolicy;
}

template <>
void Out<NCatboostCuda::TGpuTrainerFactoryKey>(IOutputStream& o, const NCatboostCuda::TGpuTrainerFactoryKey& key) {
    o.Write(ToString(key));
}
