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
                ui64 currentSize = catBoostOptions.SystemOptions->PinnedMemorySize;
                if (currentSize < cpuFeaturesSize) {
                    catBoostOptions.SystemOptions->PinnedMemorySize = cpuFeaturesSize;
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
                    options.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Simple) {
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
        config.PinnedMemorySize = systemOptions.PinnedMemorySize;
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
                                      NCatboostOptions::TCatFeatureParams& options) {
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
        NPar::ParallelFor(0, (int)featuresLayout.GetCatFeatureCount(), [&](int catFeatureIdx) {
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
                auto values = catFeatureValues.ExtractValues(&NPar::LocalExecutor());

                for (ui32 i = 0; i < currentFeatureDescription.size(); ++i) {
                    if (currentFeatureDescription[i].Type == ECtrType::Borders && options.TargetBorders->BorderCount == 1u) {
                        ui32 uniqueValues
                            = dataProvider.ObjectsData->GetQuantizedFeaturesInfo()->GetUniqueValuesCounts(TCatFeatureIdx((ui32)catFeatureIdx)).OnAll;

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

    static void SetDataDependentDefaults(const NCB::TTrainingDataProvider& dataProvider,
                                         const NCB::TTrainingDataProvider* testProvider,
                                         NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                         NCatboostOptions::TOutputFilesOptions& outputOptions,
                                         TBinarizedFeaturesManager& featuresManager) {

        bool hasTestConstTarget = true;
        bool hasTestPairs = false;
        if (testProvider) {
            hasTestConstTarget = IsConst(GetTarget(testProvider->TargetData));
            hasTestPairs = testProvider->TargetData.contains(TTargetDataSpecification(ETargetType::GroupPairwiseRanking));
        }

        UpdateUseBestModel(testProvider != nullptr, hasTestConstTarget, hasTestPairs, &outputOptions.UseBestModel);
        UpdateLearningRate(dataProvider.GetObjectCount(), outputOptions.UseBestModel.Get(), &catBoostOptions);
        UpdateBoostingTypeOption(dataProvider.GetObjectCount(),
                                 &catBoostOptions.BoostingOptions->BoostingType);

        UpdateGpuSpecificDefaults(catBoostOptions, featuresManager);
        EstimatePriors(dataProvider, featuresManager, catBoostOptions.CatFeatureParams);
        UpdateDataPartitionType(featuresManager, catBoostOptions);
        UpdatePinnedMemorySizeOption(dataProvider, testProvider, featuresManager, catBoostOptions);
    }


    THolder<TAdditiveModel<TObliviousTreeModel>> TrainModelImpl(const NCatboostOptions::TCatBoostOptions& trainCatBoostOptions,
                                                                const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                const TTrainingDataProvider& dataProvider,
                                                                const TTrainingDataProvider* testProvider,
                                                                TBinarizedFeaturesManager& featuresManager,
                                                                ui32 approxDimension,
                                                                const TMaybe<TOnEndIterationCallback>& onEndIterationCallback,
                                                                TVector<TVector<double>>* testMultiApprox, // [dim][objectIdx]
                                                                TMetricsAndTimeLeftHistory* metricsAndTimeHistory) {
        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

        if (trainCatBoostOptions.IsProfile) {
            profiler.SetDefaultProfileMode(NCudaLib::EProfileMode::ImplicitLabelSync);
        } else {
            profiler.SetDefaultProfileMode(NCudaLib::EProfileMode::NoProfile);
        }
        TGpuAwareRandom random(trainCatBoostOptions.RandomSeed);

        const auto lossFunction = trainCatBoostOptions.LossFunctionDescription->GetLossFunction();

        if (TGpuTrainerFactory::Has(lossFunction)) {
            THolder<IGpuTrainer> trainer = TGpuTrainerFactory::Construct(lossFunction);
            return trainer->TrainModel(featuresManager,
                                       trainCatBoostOptions,
                                       outputOptions,
                                       dataProvider,
                                       testProvider,
                                       random,
                                       approxDimension,
                                       onEndIterationCallback,
                                       testMultiApprox,
                                       metricsAndTimeHistory);
        } else {
            ythrow TCatboostException() << "Error: loss function is not supported for GPU learning " << lossFunction;
        }
        return nullptr; // return default to keep compiler happy
    }

    inline void CreateDirIfNotExist(const TString& path) {
        TFsPath trainDirPath(path);
        try {
            if (!path.empty() && !trainDirPath.Exists()) {
                trainDirPath.MkDir();
            }
        } catch (...) {
            ythrow TCatboostException() << "Can't create working dir: " << path;
        }
    }


    class TGPUModelTrainer: public IModelTrainer {
    public:
        void TrainModel(
            bool calcMetricsOnly,
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
            TMetricsAndTimeLeftHistory* metricsAndTimeHistory) const override
        {
            Y_UNUSED(objectiveDescriptor);
            Y_UNUSED(evalMetricDescriptor);
            Y_UNUSED(rand);
            CB_ENSURE(trainingData.Test.size() <= 1, "Multiple eval sets not supported for GPU");
            Y_VERIFY(evalResultPtrs.size() == trainingData.Test.size());

            NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::GPU);
            catBoostOptions.Load(params);

            bool saveFinalCtrsInModel
                = !calcMetricsOnly &&
                    (outputOptions.GetFinalCtrComputationMode() == EFinalCtrComputationMode::Default) &&
                    HasFeaturesForCtrs(*trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo(),
                                       catBoostOptions.CatFeatureParams.Get().OneHotMaxSize);

            TTrainingForCPUDataProviders trainingDataForFinalCtrCalculation;

            if (saveFinalCtrsInModel) {
                // do it at this stage to check before training
                trainingDataForFinalCtrCalculation
                    = trainingData.Cast<TQuantizedForCPUObjectsDataProvider>();
            }

            auto quantizedFeaturesInfo = trainingData.Learn->ObjectsData->GetQuantizedFeaturesInfo();

            TBinarizedFeaturesManager featuresManager(catBoostOptions.CatFeatureParams,
                                                      quantizedFeaturesInfo);


            NCatboostOptions::TOutputFilesOptions updatedOutputOptions = outputOptions;

            SetDataDependentDefaults(
                *trainingData.Learn,
                !trainingData.Test.empty() ? trainingData.Test[0].Get() : nullptr,
                catBoostOptions,
                updatedOutputOptions,
                featuresManager);

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
                catBoostOptions,
                updatedOutputOptions,
                *trainingData.Learn,
                !trainingData.Test.empty() ? trainingData.Test[0].Get() : nullptr,
                featuresManager,
                approxDimension,
                onEndIterationCallback,
                &rawValues,
                metricsAndTimeHistory);

            if (evalResultPtrs.size()) {
                evalResultPtrs[0]->SetRawValuesByMove(rawValues);
            }

            if (calcMetricsOnly) {
                return;
            }

            TPerfectHashedToHashedCatValuesMap perfectHashedToHashedCatValuesMap
                = quantizedFeaturesInfo->CalcPerfectHashedToHashedCatValuesMap(localExecutor);

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
                saveFinalCtrsInModel ? EFinalCtrComputationMode::Default : EFinalCtrComputationMode::Skip
            );

            coreModelToFullModelConverter.WithBinarizedDataComputedFrom(
                std::move(trainingDataForFinalCtrCalculation),
                std::move(featureCombinationToProjection),
                targetClassifiers
            ).WithPerfectHashedToHashedCatValuesMap(
                &perfectHashedToHashedCatValuesMap
            ).WithCoreModelFrom(
                modelPtr
            ).WithObjectsDataFrom(trainingData.Learn->ObjectsData);

            if (model) {
                coreModelToFullModelConverter.Do(true, model);
            } else {
                coreModelToFullModelConverter.Do(
                    updatedOutputOptions.CreateResultModelFullPath(),
                    updatedOutputOptions.GetModelFormats(),
                    updatedOutputOptions.AddFileFormatExtension()
                );
            }
        }
    };

}

TTrainerFactory::TRegistrator<NCatboostCuda::TGPUModelTrainer> GPURegistrator(ETaskType::GPU);
