#include "train.h"
#include "model_helpers.h"

#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/helpers/eval_helpers.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/cuda/ctrs/prior_estimator.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/models/oblivious_model.h>

#include <catboost/cuda/data/load_data.h>
#include <catboost/cuda/data/cat_feature_perfect_hash.h>
#include <catboost/cuda/gpu_data/pinned_memory_estimation.h>
#include <catboost/cuda/cuda_lib/devices_provider.h>
#include <catboost/cuda/cuda_lib/memory_copy_performance.h>
#include <catboost/cuda/cpu_compatibility_helpers/model_converter.h>
#include <catboost/cuda/cpu_compatibility_helpers/full_model_saver.h>
#include <catboost/cuda/cpu_compatibility_helpers/cpu_pool_based_data_provider_builder.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>

class TGPUModelTrainer: public IModelTrainer {
public:
    void TrainModel(
        const NJson::TJsonValue& params,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        TPool& learnPool,
        bool allowClearPool,
        const TPool& testPool,
        TFullModel* model,
        TEvalResult* evalResult) const override {
        Y_UNUSED(objectiveDescriptor);
        Y_UNUSED(evalMetricDescriptor);
        Y_UNUSED(allowClearPool);
        NCatboostCuda::TrainModel(params, outputOptions, learnPool, testPool, model);
        evalResult->GetRawValuesRef().resize(model->ObliviousTrees.ApproxDimension);
    }

    void TrainModel(const NCatboostOptions::TPoolLoadParams& poolLoadParams,
                    const NCatboostOptions::TOutputFilesOptions& outputOptions,
                    const NJson::TJsonValue& trainParams) const override {
        NCatboostCuda::TrainModel(poolLoadParams, outputOptions, trainParams);
    }
};

TTrainerFactory::TRegistrator<TGPUModelTrainer> GPURegistrator(ETaskType::GPU);

namespace NCatboostCuda {
    inline void UpdatePinnedMemorySizeOption(const TDataProvider& learn,
                                             const TDataProvider* test,
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
                MATRIXNET_DEBUG_LOG << "No catFeatures for ctrs found and don't look ahead is disabled. Fallback to one permutation" << Endl;
            }
            options.BoostingOptions->PermutationCount = 1;
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

    inline THolder<NCatboostOptions::TCatBoostOptions> TryToLoadSnapshotOptionsAndUpdateSeed(const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                                             NCatboostOptions::TCatBoostOptions* options) {
        auto snapshotFullPath = outputOptions.CreateSnapshotFullPath();
        if (outputOptions.SaveSnapshot() && NFs::Exists(snapshotFullPath)) {
            if (GetFileLength(snapshotFullPath) == 0) {
                MATRIXNET_WARNING_LOG << "Empty snapshot file. Something is wrong" << Endl;
                return nullptr;
            } else {
                TString jsonOptionsStr;
                TProgressHelper(ToString<ETaskType>(options->GetTaskType())).CheckedLoad(snapshotFullPath, [&](TIFStream* in) {
                    ::Load(in, jsonOptionsStr);
                });

                auto progressOptions = MakeHolder<NCatboostOptions::TCatBoostOptions>(options->GetTaskType());
                NJson::TJsonValue progressOptionsJson;
                NJson::ReadJsonTree(jsonOptionsStr, &progressOptionsJson);
                progressOptions->Load(progressOptionsJson);
                options->RandomSeed = progressOptions->RandomSeed;
                return progressOptions;
            }
        } else {
            return nullptr;
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

    static inline void EstimatePriors(const TDataProvider& dataProvider,
                                      TBinarizedFeaturesManager& featureManager,
                                      NCatboostOptions::TCatFeatureParams& options) {
        CB_ENSURE(&(featureManager.GetCatFeatureOptions()) == &options, "Error: for consistent catFeature options should be equal to one in feature manager");

        bool needSimpleCtrsPriorEstimation = NeedPriorEstimation(options.SimpleCtrs);
        const auto& borders = featureManager.GetTargetBorders();
        if (borders.size() > 1) {
            return;
        }
        auto binarizedTarget = BinarizeLine<ui8>(dataProvider.GetTargets().data(), dataProvider.GetTargets().size(), ENanMode::Forbidden, borders);

        TVector<int> catFeatureIds(dataProvider.GetCatFeatureIds().begin(), dataProvider.GetCatFeatureIds().end());
        TAdaptiveLock lock;

        //TODO(noxoomo): locks here are ugly and error prone
        NPar::ParallelFor(0, catFeatureIds.size(), [&](int i) {
            ui32 catFeature = catFeatureIds[i];
            if (!dataProvider.HasFeatureId(catFeature)) {
                return;
            }
            const ICatFeatureValuesHolder& catFeatureValues = dynamic_cast<const ICatFeatureValuesHolder&>(dataProvider.GetFeatureById(catFeature));

            bool hasPerFeatureCtr = false;

            with_lock (lock) {
                if (needSimpleCtrsPriorEstimation && !options.PerFeatureCtrs->has(catFeature)) {
                    options.PerFeatureCtrs.Get()[catFeature] = options.SimpleCtrs;
                }
                hasPerFeatureCtr = options.PerFeatureCtrs->has(catFeature);
            }

            if (hasPerFeatureCtr) {
                TVector<NCatboostOptions::TCtrDescription> currentFeatureDescription;
                with_lock (lock) {
                    currentFeatureDescription = options.PerFeatureCtrs->at(catFeature);
                }
                if (!NeedPriorEstimation(currentFeatureDescription)) {
                    return;
                }
                auto values = catFeatureValues.ExtractValues();

                for (ui32 i = 0; i < currentFeatureDescription.size(); ++i) {
                    if (currentFeatureDescription[i].Type == ECtrType::Borders && options.TargetBorders->BorderCount == 1u) {
                        TBetaPriorEstimator::TBetaPrior prior = TBetaPriorEstimator::EstimateBetaPrior(binarizedTarget.data(),
                                                                                                       values.data(), values.size(), catFeatureValues.GetUniqueValues());

                        MATRIXNET_INFO_LOG << "Estimate borders-ctr prior for feature #" << catFeature << ": " << prior.Alpha << " / " << prior.Beta << Endl;
                        currentFeatureDescription[i].Priors = {{(float)prior.Alpha, (float)(prior.Alpha + prior.Beta)}};
                    } else {
                        CB_ENSURE(currentFeatureDescription[i].PriorEstimation == EPriorEstimation::No, "Error: auto prior estimation is not available for ctr type " << currentFeatureDescription[i].Type);
                    }
                }
                with_lock (lock) {
                    options.PerFeatureCtrs.Get()[catFeature] = currentFeatureDescription;
                }
            }
        });
    }

    static void SetDataDependentDefaults(const THolder<NCatboostOptions::TCatBoostOptions>& snapshotOptions,
                                         const TDataProvider& dataProvider,
                                         const THolder<TDataProvider>& testProvider,
                                         NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                         NCatboostOptions::TOutputFilesOptions& outputOptions,
                                         TBinarizedFeaturesManager& featuresManager) {
        UpdateBoostingTypeOption(dataProvider.GetSampleCount(),
                                 &catBoostOptions.BoostingOptions->BoostingType);

        bool hasTest = testProvider.Get() != nullptr;
        bool hasTestConstTarget = true;
        if (hasTest) {
            hasTestConstTarget = IsConst(testProvider->GetTargets());
        }
        UpdateUseBestModel(hasTest, hasTestConstTarget, &outputOptions.UseBestModel);

        UpdateGpuSpecificDefaults(catBoostOptions, featuresManager);
        if (snapshotOptions.Get() == nullptr) {
            EstimatePriors(dataProvider, featuresManager, catBoostOptions.CatFeatureParams);
        } else {
            catBoostOptions.CatFeatureParams = snapshotOptions->CatFeatureParams;
        }
        UpdateDataPartitionType(featuresManager, catBoostOptions);
        UpdatePinnedMemorySizeOption(dataProvider, testProvider.Get(), featuresManager, catBoostOptions);

        // TODO(nikitxskv): Remove it when the l2 normalization will be added.
        UpdateLeavesEstimation(!dataProvider.IsTrivialWeights(), &catBoostOptions);
    }

    inline TFullModel TrainModelImpl(const NCatboostOptions::TCatBoostOptions& trainCatBoostOptions,
                                     const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                     const TDataProvider& dataProvider,
                                     const TDataProvider* testProvider,
                                     TBinarizedFeaturesManager& featuresManager) {
        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

        if (trainCatBoostOptions.IsProfile) {
            profiler.SetDefaultProfileMode(NCudaLib::EProfileMode::ImplicitLabelSync);
        } else {
            profiler.SetDefaultProfileMode(NCudaLib::EProfileMode::NoProfile);
        }
        TRandom random(trainCatBoostOptions.RandomSeed);

        THolder<TAdditiveModel<TObliviousTreeModel>> model;
        const bool storeCatFeaturesInPinnedMemory = trainCatBoostOptions.DataProcessingOptions->GpuCatFeaturesStorage == EGpuCatFeaturesStorage::CpuPinnedMemory;

        const auto lossFunction = trainCatBoostOptions.LossFunctionDescription->GetLossFunction();

        if (TGpuTrainerFactory::Has(lossFunction)) {
            THolder<IGpuTrainer> trainer = TGpuTrainerFactory::Construct(lossFunction);
            model = trainer->TrainModel(featuresManager, trainCatBoostOptions, outputOptions, dataProvider, testProvider, random, storeCatFeaturesInPinnedMemory);
        } else {
            ythrow TCatboostException() << "Error: loss function is not supported for GPU learning " << lossFunction;
        }

        TFullModel result = ConvertToCoreModel(featuresManager,
                                               dataProvider,
                                               *model);
        {
            NJson::TJsonValue options(NJson::EJsonValueType::JSON_MAP);
            trainCatBoostOptions.Save(&options);
            result.ModelInfo["params"] = ToString(options);
        }
        return result;
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

    TFullModel TrainModel(const NCatboostOptions::TCatBoostOptions& trainCatBoostOptions,
                          const NCatboostOptions::TOutputFilesOptions& outputOptions,
                          const TDataProvider& dataProvider,
                          const TDataProvider* testProvider,
                          TBinarizedFeaturesManager& featuresManager) {
        SetLogingLevel(trainCatBoostOptions.LoggingLevel);
        CreateDirIfNotExist(outputOptions.GetTrainDir());
        auto deviceRequestConfig = CreateDeviceRequestConfig(trainCatBoostOptions);
        auto stopCudaManagerGuard = StartCudaManager(deviceRequestConfig,
                                                     trainCatBoostOptions.LoggingLevel);

        return TrainModelImpl(trainCatBoostOptions, outputOptions, dataProvider, testProvider, featuresManager);
    }

    void TrainModel(const NJson::TJsonValue& params,
                    const NCatboostOptions::TOutputFilesOptions& outputOptions,
                    TPool& learnPool,
                    const TPool& testPool,
                    TFullModel* model) {
        TString outputModelPath = outputOptions.CreateResultModelFullPath();
        NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::GPU);
        catBoostOptions.Load(params);
        THolder<TCatboostOptions> snapshotOptions = TryToLoadSnapshotOptionsAndUpdateSeed(outputOptions,
                                                                                          &catBoostOptions);
        MATRIXNET_INFO_LOG << "Random seed " << catBoostOptions.RandomSeed << Endl;
        SetLogingLevel(catBoostOptions.LoggingLevel);
        TDataProvider dataProvider;
        THolder<TDataProvider> testData;
        if (testPool.Docs.GetDocCount()) {
            testData = MakeHolder<TDataProvider>();
        }

        TVector<ui64> indices(learnPool.Docs.GetDocCount());
        std::iota(indices.begin(), indices.end(), 0);

        ui64 minTimestamp = *MinElement(learnPool.Docs.Timestamp.begin(), learnPool.Docs.Timestamp.end());
        ui64 maxTimestamp = *MaxElement(learnPool.Docs.Timestamp.begin(), learnPool.Docs.Timestamp.end());
        if (minTimestamp != maxTimestamp) {
            indices = CreateOrderByKey(learnPool.Docs.Timestamp);
            catBoostOptions.DataProcessingOptions->HasTimeFlag = true;
        }

        const ui32 numThreads = catBoostOptions.SystemOptions->NumThreads;
        if (NPar::LocalExecutor().GetThreadCount() < (int)numThreads) {
            NPar::LocalExecutor().RunAdditionalThreads(numThreads - NPar::LocalExecutor().GetThreadCount());
        }

        bool hasQueries = false;
        for (ui32 i = 0; i < learnPool.Docs.QueryId.size(); ++i) {
            if (learnPool.Docs.QueryId[i] != learnPool.Docs.QueryId[0]) {
                hasQueries = true;
                break;
            }
        }

        if (!catBoostOptions.DataProcessingOptions->HasTimeFlag) {
            const ui64 shuffleSeed = catBoostOptions.RandomSeed;
            if (hasQueries) {
                QueryConsistentShuffle(shuffleSeed, 1u, learnPool.Docs.QueryId, &indices);
            } else {
                Shuffle(shuffleSeed, 1u, indices.size(), &indices);
            }
        } else {
            dataProvider.SetHasTimeFlag(true);
        }
        auto& localExecutor = NPar::LocalExecutor();
        ::ApplyPermutation(InvertPermutation(indices), &learnPool, &localExecutor);
        auto permutationGuard = Finally([&] { ::ApplyPermutation(indices, &learnPool, &localExecutor); });

        auto ignoredFeatures = catBoostOptions.DataProcessingOptions->IgnoredFeatures;

        TBinarizedFeaturesManager featuresManager(catBoostOptions.CatFeatureParams,
                                                  catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization);

        {
            CB_ENSURE(learnPool.Docs.GetDocCount(), "Error: empty learn pool");
            TCpuPoolBasedDataProviderBuilder builder(featuresManager, hasQueries, learnPool, false, dataProvider);
            builder.AddIgnoredFeatures(ignoredFeatures.Get())
                .SetClassesWeights(catBoostOptions.DataProcessingOptions->ClassWeights)
                .Finish(numThreads);
        }

        if (testData != nullptr) {
            TCpuPoolBasedDataProviderBuilder builder(featuresManager, hasQueries, testPool, true, *testData);
            builder.AddIgnoredFeatures(ignoredFeatures.Get())
                .SetClassesWeights(catBoostOptions.DataProcessingOptions->ClassWeights)
                .Finish(numThreads);
        }

        auto outputOptionsFinal = outputOptions;
        SetDataDependentDefaults(snapshotOptions,
                                 dataProvider,
                                 testData,
                                 catBoostOptions,
                                 outputOptionsFinal,
                                 featuresManager);


        auto coreModel = TrainModel(catBoostOptions, outputOptionsFinal, dataProvider, testData.Get(), featuresManager);
        auto targetClassifiers = CreateTargetClassifiers(featuresManager);
        if (model == nullptr) {
            CB_ENSURE(!outputModelPath.Size(), "Error: Model and output path are empty");
            MakeFullModel(std::move(coreModel),
                          learnPool,
                          targetClassifiers,
                          numThreads,
                          outputModelPath);
        } else {
            MakeFullModel(std::move(coreModel),
                          learnPool,
                          targetClassifiers,
                          numThreads,
                          model);
        }
    }

    void TrainModel(const NCatboostOptions::TPoolLoadParams& poolLoadOptions,
                    const NCatboostOptions::TOutputFilesOptions& outputOptions,
                    const NJson::TJsonValue& jsonOptions) {
        auto catBoostOptions = NCatboostOptions::LoadOptions(jsonOptions);
        auto snapshotOptions = TryToLoadSnapshotOptionsAndUpdateSeed(outputOptions, &catBoostOptions);
        MATRIXNET_INFO_LOG << "Random seed " << catBoostOptions.RandomSeed << Endl;

        SetLogingLevel(catBoostOptions.LoggingLevel);
        const auto resultModelPath = outputOptions.CreateResultModelFullPath();
        TString coreModelPath = TStringBuilder() << resultModelPath << ".core";

        const int numThreads = catBoostOptions.SystemOptions->NumThreads;
        if (NPar::LocalExecutor().GetThreadCount() < numThreads) {
            NPar::LocalExecutor().RunAdditionalThreads(numThreads - NPar::LocalExecutor().GetThreadCount());
        }

        TVector<TTargetClassifier> targetClassifiers;
        {
            TBinarizedFeaturesManager featuresManager(catBoostOptions.CatFeatureParams,
                                                      catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization);

            TDataProvider dataProvider;
            THolder<TDataProvider> testProvider;

            {
                TDataProviderBuilder dataProviderBuilder(featuresManager,
                                                         dataProvider,
                                                         false,
                                                         numThreads);

                const auto& ignoredFeatures = catBoostOptions.DataProcessingOptions->IgnoredFeatures;
                dataProviderBuilder
                    .AddIgnoredFeatures(ignoredFeatures.Get());

                if (!catBoostOptions.DataProcessingOptions->HasTimeFlag) {
                    dataProviderBuilder.SetShuffleFlag(true, catBoostOptions.RandomSeed);
                } else {
                    dataProvider.SetHasTimeFlag(true);
                }
                dataProviderBuilder
                    .SetClassesWeights(catBoostOptions.DataProcessingOptions->ClassWeights);

                {
                    MATRIXNET_DEBUG_LOG << "Loading features..." << Endl;
                    auto start = Now();
                    ReadPool(poolLoadOptions.CdFile,
                             poolLoadOptions.LearnFile,
                             poolLoadOptions.PairsFile,
                             poolLoadOptions.IgnoredFeatures,
                             true,
                             poolLoadOptions.Delimiter,
                             poolLoadOptions.HasHeader,
                             catBoostOptions.DataProcessingOptions->ClassNames,
                             &NPar::LocalExecutor(),
                             &dataProviderBuilder);
                    MATRIXNET_DEBUG_LOG << "Loading features time: " << (Now() - start).Seconds() << Endl;
                }

                if (poolLoadOptions.TestFile) {
                    MATRIXNET_DEBUG_LOG << "Loading test..." << Endl;
                    testProvider.Reset(new TDataProvider());
                    TDataProviderBuilder testBuilder(featuresManager,
                                                     *testProvider,
                                                     true,
                                                     numThreads);
                    testBuilder
                        .AddIgnoredFeatures(ignoredFeatures.Get())
                        .SetShuffleFlag(false)
                        .SetClassesWeights(catBoostOptions.DataProcessingOptions->ClassWeights);

                    ReadPool(poolLoadOptions.CdFile,
                             poolLoadOptions.TestFile,
                             poolLoadOptions.TestPairsFile,
                             poolLoadOptions.IgnoredFeatures,
                             true,
                             poolLoadOptions.Delimiter,
                             poolLoadOptions.HasHeader,
                             catBoostOptions.DataProcessingOptions->ClassNames,
                             &NPar::LocalExecutor(),
                             &testBuilder);
                }
            }

            featuresManager.UnloadCatFeaturePerfectHashFromRam();
            auto outputOptionsFinal = outputOptions;
            SetDataDependentDefaults(snapshotOptions, dataProvider, testProvider, catBoostOptions, outputOptionsFinal, featuresManager);

            {
                auto coreModel = TrainModel(catBoostOptions, outputOptionsFinal, dataProvider, testProvider.Get(), featuresManager);
                TOFStream modelOutput(coreModelPath);
                coreModel.Save(&modelOutput);
            }
            targetClassifiers = CreateTargetClassifiers(featuresManager);
        }

        MakeFullModel(coreModelPath,
                      poolLoadOptions,
                      catBoostOptions.DataProcessingOptions->ClassNames,
                      targetClassifiers,
                      numThreads,
                      resultModelPath);
    }
}
