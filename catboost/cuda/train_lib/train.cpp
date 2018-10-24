#include "train.h"

#include <catboost/cuda/cpu_compatibility_helpers/model_converter.h>
#include <catboost/cuda/cpu_compatibility_helpers/cpu_pool_based_data_provider_builder.h>
#include <catboost/cuda/ctrs/prior_estimator.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_lib/devices_provider.h>
#include <catboost/cuda/cuda_lib/memory_copy_performance.h>
#include <catboost/cuda/cuda_util/gpu_random.h>
#include <catboost/cuda/data/load_data.h>
#include <catboost/cuda/data/cat_feature_perfect_hash.h>
#include <catboost/cuda/gpu_data/pinned_memory_estimation.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/models/oblivious_model.h>

#include <catboost/libs/algo/full_model_saver.h>
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/eval_result/eval_helpers.h>
#include <catboost/libs/fstr/output_fstr.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/quantization/utils.h>
#include <catboost/libs/train_lib/preprocess.h>

#include <library/json/json_prettifier.h>

#include <util/generic/scope.h>
#include <util/system/info.h>

class TGPUModelTrainer: public IModelTrainer {
public:
    void TrainModel(
        const NJson::TJsonValue& params,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const TClearablePoolPtrs& pools,
        TFullModel* model,
        const TVector<TEvalResult*>& evalResultPtrs,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory) const override
    {
        Y_UNUSED(objectiveDescriptor);
        Y_UNUSED(evalMetricDescriptor);
        CB_ENSURE(pools.Test.size() <= 1, "Multiple eval sets not supported for GPU");
        Y_VERIFY(evalResultPtrs.size() == pools.Test.size());

        NCatboostCuda::TrainModel(params, outputOptions, *pools.Learn, pools.Test.size() ? *pools.Test[0] : TPool(), model, metricsAndTimeHistory);
        if (evalResultPtrs.size()) {
            evalResultPtrs[0]->GetRawValuesRef().resize(model->ObliviousTrees.ApproxDimension);
        }
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

    inline void TryUpdateSeedFromSnapshot(const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                NJson::TJsonValue* updatedParams) {
        UpdateUndefinedRandomSeed(ETaskType::GPU, outputOptions, updatedParams, [&](IInputStream* in, TString& params) {
            ::Load(in, params);
        });
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
        auto binarizedTarget = NCB::BinarizeLine<ui8>(dataProvider.GetTargets(), ENanMode::Forbidden, borders);

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

                        CATBOOST_INFO_LOG << "Estimate borders-ctr prior for feature #" << catFeature << ": " << prior.Alpha << " / " << prior.Beta << Endl;
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

    static void SetDataDependentDefaults(const TDataProvider& dataProvider,
                                         const THolder<TDataProvider>& testProvider,
                                         NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                         NCatboostOptions::TOutputFilesOptions& outputOptions,
                                         TBinarizedFeaturesManager& featuresManager) {

        bool hasTest = testProvider.Get() != nullptr;
        bool hasTestConstTarget = true;
        if (hasTest) {
            hasTestConstTarget = IsConst(testProvider->GetTargets());
        }

        UpdateUseBestModel(hasTest, hasTestConstTarget, &outputOptions.UseBestModel);
        UpdateLearningRate(dataProvider.GetSampleCount(), outputOptions.UseBestModel.Get(), &catBoostOptions);
        UpdateBoostingTypeOption(dataProvider.GetSampleCount(),
                                 &catBoostOptions.BoostingOptions->BoostingType);

        UpdateGpuSpecificDefaults(catBoostOptions, featuresManager);
        EstimatePriors(dataProvider, featuresManager, catBoostOptions.CatFeatureParams);
        UpdateDataPartitionType(featuresManager, catBoostOptions);
        UpdatePinnedMemorySizeOption(dataProvider, testProvider.Get(), featuresManager, catBoostOptions);

        // TODO(nikitxskv): Remove it when the l2 normalization will be added.
        UpdateLeavesEstimation(!dataProvider.IsTrivialWeights(), &catBoostOptions);
    }

    static inline bool NeedShuffle(const ui64 catFeatureCount, const ui64 docCount, const NCatboostOptions::TCatBoostOptions& catBoostOptions) {
        if (catBoostOptions.DataProcessingOptions->HasTimeFlag) {
            return false;
        }

        if (catFeatureCount == 0) {
            auto boostingType = catBoostOptions.BoostingOptions->BoostingType;
            UpdateBoostingTypeOption(docCount,
                                     &boostingType);
            if (boostingType ==  EBoostingType::Ordered) {
                return true;
            } else {
                return false;
            }
        } else {
            return true;
        }
    }

    inline TFullModel TrainModelImpl(const NCatboostOptions::TCatBoostOptions& trainCatBoostOptions,
                                     const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                     const TDataProvider& dataProvider,
                                     const TDataProvider* testProvider,
                                     TBinarizedFeaturesManager& featuresManager,
                                     TMetricsAndTimeLeftHistory* metricsAndTimeHistory) {
        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

        if (trainCatBoostOptions.IsProfile) {
            profiler.SetDefaultProfileMode(NCudaLib::EProfileMode::ImplicitLabelSync);
        } else {
            profiler.SetDefaultProfileMode(NCudaLib::EProfileMode::NoProfile);
        }
        TGpuAwareRandom random(trainCatBoostOptions.RandomSeed);

        THolder<TAdditiveModel<TObliviousTreeModel>> model;

        const auto lossFunction = trainCatBoostOptions.LossFunctionDescription->GetLossFunction();

        if (TGpuTrainerFactory::Has(lossFunction)) {
            THolder<IGpuTrainer> trainer = TGpuTrainerFactory::Construct(lossFunction);
            model = trainer->TrainModel(featuresManager, trainCatBoostOptions, outputOptions, dataProvider, testProvider, random, metricsAndTimeHistory);
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
            for (const auto& keyValue : trainCatBoostOptions.Metadata.Get().GetMap()) {
                result.ModelInfo[keyValue.first] = keyValue.second.GetString();
            }
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
                          TBinarizedFeaturesManager& featuresManager,
                          TMetricsAndTimeLeftHistory* metricsAndTimeHistory) {
        SetLoggingLevel(trainCatBoostOptions.LoggingLevel);
        CreateDirIfNotExist(outputOptions.GetTrainDir());
        auto deviceRequestConfig = CreateDeviceRequestConfig(trainCatBoostOptions);
        auto stopCudaManagerGuard = StartCudaManager(deviceRequestConfig,
                                                     trainCatBoostOptions.LoggingLevel);

        const ui32 workingThreads = NPar::LocalExecutor().GetThreadCount() + 1;
        if (workingThreads < trainCatBoostOptions.SystemOptions->NumThreads) {
            NPar::LocalExecutor().RunAdditionalThreads(trainCatBoostOptions.SystemOptions->NumThreads - workingThreads);
        }
        return TrainModelImpl(trainCatBoostOptions, outputOptions, dataProvider, testProvider, featuresManager, metricsAndTimeHistory);
    }



    void TrainModel(const NJson::TJsonValue& params,
                    const NCatboostOptions::TOutputFilesOptions& outputOptions,
                    TPool& learnPool,
                    const TPool& testPool,
                    TFullModel* model,
                    TMetricsAndTimeLeftHistory* metricsAndTimeHistory) {
        TString outputModelPath = outputOptions.CreateResultModelFullPath();
        NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::GPU);
        NJson::TJsonValue updatedParams = params;
        TryUpdateSeedFromSnapshot(outputOptions, &updatedParams);
        catBoostOptions.Load(updatedParams);
        CATBOOST_INFO_LOG << "Random seed " << catBoostOptions.RandomSeed << Endl;
        SetLoggingLevel(catBoostOptions.LoggingLevel);
        TDataProvider dataProvider;
        THolder<TDataProvider> testData;
        if (testPool.Docs.GetDocCount()) {
            testData = MakeHolder<TDataProvider>();
        }

        TVector<ui64> indices(learnPool.Docs.GetDocCount());
        std::iota(indices.begin(), indices.end(), 0);

        ui64 minTimestamp = *MinElement(learnPool.Docs.Timestamp.begin(), learnPool.Docs.Timestamp.end());
        ui64 maxTimestamp = *MaxElement(learnPool.Docs.Timestamp.begin(), learnPool.Docs.Timestamp.end());

        const bool hasTimestamps = minTimestamp != maxTimestamp;
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


        const bool needReorder = hasTimestamps || NeedShuffle(learnPool.CatFeatures.size(), learnPool.Docs.GetDocCount(), catBoostOptions);
        if (!catBoostOptions.DataProcessingOptions->HasTimeFlag) {
            if (needReorder) {
                const ui64 shuffleSeed = catBoostOptions.RandomSeed;
                if (hasQueries) {
                    QueryConsistentShuffle(shuffleSeed, 1u, learnPool.Docs.QueryId, &indices);
                } else {
                    Shuffle(shuffleSeed, 1u, indices.size(), &indices);
                }
            }
        } else {
            dataProvider.SetHasTimeFlag(true);
        }

        auto& localExecutor = NPar::LocalExecutor();

        if (needReorder) {
            ::ApplyPermutation(InvertPermutation(indices), &learnPool, &localExecutor);
        }
        Y_DEFER {
            if (needReorder) {
                ::ApplyPermutation(indices, &learnPool, &localExecutor);
            }
        };

        auto ignoredFeatures = catBoostOptions.DataProcessingOptions->IgnoredFeatures;

        TBinarizedFeaturesManager featuresManager(catBoostOptions.CatFeatureParams,
                                                  catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization);

        TSimpleSharedPtr<TClassificationTargetHelper> targetHelper;

        if (IsClassificationObjective(catBoostOptions.LossFunctionDescription->GetLossFunction())) {
            targetHelper = new TClassificationTargetHelper(catBoostOptions);
        }

        {
            CB_ENSURE(learnPool.Docs.GetDocCount(), "Error: empty learn pool");
            TCpuPoolBasedDataProviderBuilder builder(
                    featuresManager,
                    hasQueries,
                    learnPool,
                    false,
                    catBoostOptions.LossFunctionDescription,
                    catBoostOptions.RandomSeed,
                    dataProvider);

            builder.AddIgnoredFeatures(ignoredFeatures.Get())
                .SetTargetHelper(targetHelper)
                .Finish(numThreads);
        }

        if (testData != nullptr) {
            TCpuPoolBasedDataProviderBuilder builder(
                    featuresManager,
                    hasQueries,
                    testPool,
                    true,
                    catBoostOptions.LossFunctionDescription,
                    catBoostOptions.RandomSeed,
                    *testData);

            builder.AddIgnoredFeatures(ignoredFeatures.Get())
                .SetTargetHelper(targetHelper)
                .Finish(numThreads);
        }

        auto outputOptionsFinal = outputOptions;
        SetDataDependentDefaults(dataProvider,
                                 testData,
                                 catBoostOptions,
                                 outputOptionsFinal,
                                 featuresManager);

        auto coreModel = TrainModel(catBoostOptions, outputOptionsFinal, dataProvider, testData.Get(), featuresManager, metricsAndTimeHistory);
        auto targetClassifiers = CreateTargetClassifiers(featuresManager);

        NCB::TCoreModelToFullModelConverter coreModelToFullModelConverter(
            numThreads,
            outputOptions.GetFinalCtrComputationMode(),
            ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get()),
            /*ctrLeafCountLimit*/ Max<ui64>(),
            /*storeAllSimpleCtrs*/ false,
            catBoostOptions.CatFeatureParams
        );

        coreModelToFullModelConverter.WithCoreModelFrom(&coreModel);

        coreModelToFullModelConverter.WithBinarizedDataComputedFrom(
            TClearablePoolPtrs(learnPool, {&testPool}),
            targetClassifiers
        );

        if (model == nullptr) {
            CB_ENSURE(!outputModelPath.Size(), "Error: Model and output path are empty");
            coreModelToFullModelConverter.Do(outputModelPath);
        } else {
            coreModelToFullModelConverter.Do(model, true);
        }
    }

    void TrainModel(const NCatboostOptions::TPoolLoadParams& poolLoadOptions,
                    const NCatboostOptions::TOutputFilesOptions& outputOptions,
                    const NJson::TJsonValue& jsonOptions) {
        NJson::TJsonValue updatedOptions = jsonOptions;
        TryUpdateSeedFromSnapshot(outputOptions, &updatedOptions);
        auto catBoostOptions = NCatboostOptions::LoadOptions(updatedOptions);
        CATBOOST_INFO_LOG << "Random seed " << catBoostOptions.RandomSeed << Endl;

        SetLoggingLevel(catBoostOptions.LoggingLevel);
        const auto resultModelPath = outputOptions.CreateResultModelFullPath();
        TString coreModelPath = TStringBuilder() << resultModelPath << ".core";

        const int numThreads = catBoostOptions.SystemOptions->NumThreads;
        if (NPar::LocalExecutor().GetThreadCount() < numThreads) {
            NPar::LocalExecutor().RunAdditionalThreads(numThreads - NPar::LocalExecutor().GetThreadCount());
        }

        TVector<TTargetClassifier> targetClassifiers;
        //will be set to skip if pool without categorical features
        EFinalCtrComputationMode ctrComputationMode = outputOptions.GetFinalCtrComputationMode();
        if (poolLoadOptions.LearnSetPath.Scheme == "quantized") {
            // TODO(yazevnul): quantized pool do not support categorical features yet
            ctrComputationMode = EFinalCtrComputationMode::Skip;
        }
        {
            TBinarizedFeaturesManager featuresManager(catBoostOptions.CatFeatureParams,
                                                      catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization);

            TDataProvider dataProvider;
            THolder<TDataProvider> testProvider;

            TBinarizedFloatFeaturesMetaInfo binarizedFloatFeaturesInfo;
            if (poolLoadOptions.BordersFile.size()) {
                binarizedFloatFeaturesInfo = LoadBordersFromFromFileInMatrixnetFormat(poolLoadOptions.BordersFile);
            }

            {
                TDataProviderBuilder dataProviderBuilder(featuresManager,
                                                         dataProvider,
                                                         false,
                                                         numThreads);

                dataProviderBuilder.SetBinarizedFeaturesMetaInfo(binarizedFloatFeaturesInfo);

                const auto& ignoredFeatures = catBoostOptions.DataProcessingOptions->IgnoredFeatures;
                dataProviderBuilder
                    .AddIgnoredFeatures(ignoredFeatures.Get());

                if (!catBoostOptions.DataProcessingOptions->HasTimeFlag) {
                    dataProviderBuilder.SetShuffleFlag(true, catBoostOptions.RandomSeed);
                } else {
                    dataProvider.SetHasTimeFlag(true);
                }



                TSimpleSharedPtr<TClassificationTargetHelper> targetHelper;
                if (IsClassificationObjective(catBoostOptions.LossFunctionDescription->GetLossFunction())) {
                    targetHelper = new TClassificationTargetHelper(catBoostOptions);
                    dataProviderBuilder.SetTargetHelper(targetHelper);
                }


                {
                    NCB::TTargetConverter targetConverter = NCB::MakeTargetConverter(catBoostOptions);
                    CATBOOST_DEBUG_LOG << "Loading features..." << Endl;
                    auto start = Now();
                    NCatboostCuda::ReadPool(
                        poolLoadOptions.LearnSetPath,
                        poolLoadOptions.PairsFilePath,
                        poolLoadOptions.GroupWeightsFilePath,
                        poolLoadOptions.DsvPoolFormatParams,
                        poolLoadOptions.IgnoredFeatures,
                        true,
                        &targetConverter,
                        &NPar::LocalExecutor(),
                        &dataProviderBuilder);
                    CATBOOST_DEBUG_LOG << "Loading features time: " << (Now() - start).Seconds() << Endl;
                }
                const auto& lossFunctionDescription = catBoostOptions.LossFunctionDescription.Get();
                if (IsPairLogit(lossFunctionDescription.GetLossFunction()) && dataProvider.GetPairs().empty()) {
                    CB_ENSURE(
                            !dataProvider.GetTargets().empty(),
                            "Pool labels are not provided. Cannot generate pairs."
                    );

                    CATBOOST_WARNING_LOG << "No pairs provided for learn dataset. "
                                          << "Trying to generate pairs using dataset labels." << Endl;
                    dataProviderBuilder.GeneratePairs(lossFunctionDescription);
                }

                if (outputOptions.NeedSaveBorders()) {
                    dataProvider.DumpBordersToFileInMatrixnetFormat(outputOptions.CreateOutputBordersFullPath());
                }

                if (poolLoadOptions.TestSetPaths.size() > 0) {
                    NCB::TTargetConverter targetConverter = NCB::MakeTargetConverter(catBoostOptions);
                    CB_ENSURE(poolLoadOptions.TestSetPaths.size() == 1, "Multiple eval sets not supported for GPU");
                    CATBOOST_DEBUG_LOG << "Loading test..." << Endl;
                    testProvider.Reset(new TDataProvider());
                    TDataProviderBuilder testBuilder(featuresManager,
                                                     *testProvider,
                                                     true,
                                                     numThreads);

                    testBuilder
                        .SetBinarizedFeaturesMetaInfo(binarizedFloatFeaturesInfo)
                        .AddIgnoredFeatures(ignoredFeatures.Get())
                        .SetShuffleFlag(false)
                        .SetTargetHelper(targetHelper);


                    NCatboostCuda::ReadPool(
                        poolLoadOptions.TestSetPaths[0],
                        poolLoadOptions.TestPairsFilePath,
                        poolLoadOptions.TestGroupWeightsFilePath,
                        poolLoadOptions.DsvPoolFormatParams,
                        poolLoadOptions.IgnoredFeatures,
                        true,
                        &targetConverter,
                        &NPar::LocalExecutor(),
                        &testBuilder);
                    if (IsPairLogit(lossFunctionDescription.GetLossFunction()) && testProvider.Get()->GetPairs().empty()) {
                        CB_ENSURE(
                                !testProvider.Get()->GetTargets().empty(),
                                "Pool labels are not provided. Cannot generate pairs."
                        );

                        CATBOOST_WARNING_LOG << "No pairs provided for test dataset. "
                                              << "Trying to generate pairs using dataset labels." << Endl;
                        testBuilder.GeneratePairs(lossFunctionDescription);
                    }
                }
            }

            featuresManager.UnloadCatFeaturePerfectHashFromRam();
            auto outputOptionsFinal = outputOptions;
            SetDataDependentDefaults(dataProvider, testProvider, catBoostOptions, outputOptionsFinal, featuresManager);

            {
                auto coreModel = TrainModel(catBoostOptions, outputOptionsFinal, dataProvider, testProvider.Get(), featuresManager, nullptr);
                if (coreModel.GetUsedCatFeaturesCount() == 0) {
                    ctrComputationMode = EFinalCtrComputationMode::Skip;
                }
                TOFStream modelOutput(coreModelPath);
                coreModel.Save(&modelOutput);
            }
            targetClassifiers = CreateTargetClassifiers(featuresManager);
        }

        NCB::TCoreModelToFullModelConverter(
            numThreads,
            ctrComputationMode,
            ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get()),
            /*ctrLeafCountLimit*/ Max<ui64>(),
            /*storeAllSimpleCtrs*/ false,
            catBoostOptions.CatFeatureParams
        ).WithCoreModelFrom(
            coreModelPath
        ).WithBinarizedDataComputedFrom(
            poolLoadOptions,
            catBoostOptions.DataProcessingOptions->ClassNames,
            targetClassifiers
        ).Do(
            resultModelPath
        );

        const auto fstrRegularFileName = outputOptions.CreateFstrRegularFullPath();
        const auto fstrInternalFileName = outputOptions.CreateFstrIternalFullPath();
        const bool needFstr = !fstrInternalFileName.empty() || !fstrRegularFileName.empty();
        if (needFstr) {
            TPool emptyPool;
            TFullModel model = ReadModel(resultModelPath);
            CalcAndOutputFstr(model, &emptyPool, &fstrRegularFileName, &fstrInternalFileName);
        }
        const TString trainingOptionsFileName = outputOptions.CreateTrainingOptionsFullPath();
        if (!trainingOptionsFileName.empty()) {
            TOFStream trainingOptionsFile(trainingOptionsFileName);
            trainingOptionsFile.Write(NJson::PrettifyJson(ToString(catBoostOptions)));
        }
    }
}
