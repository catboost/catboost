#include "train.h"
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/helpers/eval_helpers.h>
#include <catboost/cuda/ctrs/prior_estimator.h>

class TGPUModelTrainer: public IModelTrainer
{
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
            TEvalResult* evalResult) const override
    {
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

TTrainerFactory::TRegistrator <TGPUModelTrainer> GPURegistrator(ETaskType::GPU);

namespace NCatboostCuda
{

    inline void CreateAndSetCudaConfig(const NCatboostOptions::TCatBoostOptions& options)
    {
        NCudaLib::TCudaApplicationConfig config;
        const auto& systemOptions = options.SystemOptions.Get();
        config.DeviceConfig = systemOptions.Devices;
        config.PinnedMemorySize = systemOptions.PinnedMemorySize;
        config.GpuMemoryPartByWorker = systemOptions.GpuRamPart;
        NCudaLib::SetApplicationConfig(config);
    }

    inline void CheckForSnapshotAndReloadOptions(const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                 NCatboostOptions::TCatBoostOptions* options)
    {
        auto snapshotFullPath = outputOptions.CreateSnapshotFullPath();
        if (outputOptions.SaveSnapshot() && NFs::Exists(snapshotFullPath)) {
            if (GetFileLength(snapshotFullPath) == 0) {
                MATRIXNET_WARNING_LOG << "Empty snapshot file. Something is wrong" << Endl;
            } else {
                TString jsonOptions;
                TProgressHelper(ToString<ETaskType>(options->GetTaskType())).CheckedLoad(
                        snapshotFullPath, [&](TIFStream* in)
                        {
                            ::Load(in, jsonOptions);
                        });
                options->Load(jsonOptions);
            }
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
            const TCatFeatureValuesHolder& catFeatureValues = dynamic_cast<const TCatFeatureValuesHolder&>(dataProvider.GetFeatureById(catFeature));

            bool hasPerFeatureCtr = false;

            with_lock(lock) {
                if (needSimpleCtrsPriorEstimation && !options.PerFeatureCtrs->has(catFeature)) {
                    options.PerFeatureCtrs.Get()[catFeature] = options.SimpleCtrs;
                }
                hasPerFeatureCtr = options.PerFeatureCtrs->has(catFeature);
            }

            if (hasPerFeatureCtr) {
                TVector<NCatboostOptions::TCtrDescription> currentFeatureDescription;
                with_lock(lock) {
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


    inline TFullModel TrainModelImpl(const NCatboostOptions::TCatBoostOptions& trainCatBoostOptions,
                                     const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                     const TDataProvider& dataProvider,
                                     const TDataProvider* testProvider,
                                     TBinarizedFeaturesManager& featuresManager)
    {
        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

        if (trainCatBoostOptions.IsProfile)
        {
            profiler.SetDefaultProfileMode(NCudaLib::EProfileMode::ImplicitLabelSync);
        } else
        {
            profiler.SetDefaultProfileMode(NCudaLib::EProfileMode::NoProfile);
        }
        TRandom random(trainCatBoostOptions.RandomSeed);

        THolder<TAdditiveModel<TObliviousTreeModel>> model;
        const bool storeCatFeaturesInPinnedMemory = trainCatBoostOptions.DataProcessingOptions->GpuCatFeaturesStorage == EGpuCatFeaturesStorage::CpuPinnedMemory;

        const auto lossFunction = trainCatBoostOptions.LossFunctionDescription->GetLossFunction();
        switch (lossFunction)
        {
            case ELossFunction::RMSE:
            {
                model = Train<TL2>(featuresManager,
                                   trainCatBoostOptions,
                                   outputOptions,
                                   dataProvider,
                                   testProvider,
                                   random,
                                   storeCatFeaturesInPinnedMemory);
                break;
            }
            case ELossFunction::CrossEntropy:
            {
                model = Train<TCrossEntropy>(featuresManager,
                                             trainCatBoostOptions,
                                             outputOptions,
                                             dataProvider,
                                             testProvider,
                                             random,
                                             storeCatFeaturesInPinnedMemory);
                break;
            }
            case  ELossFunction::Logloss:
            {
                model = Train<TLogloss>(featuresManager,
                                        trainCatBoostOptions,
                                        outputOptions,
                                        dataProvider,
                                        testProvider,
                                        random,
                                        storeCatFeaturesInPinnedMemory);
                break;
            }
            default: {
                ythrow TCatboostException() << "Error: loss function is not supported for GPU learning " << lossFunction;
            }
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



    TFullModel TrainModel(const NCatboostOptions::TCatBoostOptions& trainCatBoostOptions,
                          const NCatboostOptions::TOutputFilesOptions& outputOptions,
                          const TDataProvider& dataProvider,
                          const TDataProvider* testProvider,
                          TBinarizedFeaturesManager& featuresManager)
    {
        std::promise<TFullModel> resultPromise;
        std::future<TFullModel> resultFuture = resultPromise.get_future();
        std::thread thread([&]()
                           {
                               SetLogingLevel(trainCatBoostOptions.LoggingLevel);
                               CreateAndSetCudaConfig(trainCatBoostOptions);

                               StartCudaManager(trainCatBoostOptions.LoggingLevel);
                               try
                               {
                                   if (NCudaLib::GetCudaManager().GetDeviceCount() > 1)
                                   {
                                       NCudaLib::GetLatencyAndBandwidthStats<NCudaLib::CudaDevice, NCudaLib::CudaHost>();
                                       NCudaLib::GetLatencyAndBandwidthStats<NCudaLib::CudaDevice, NCudaLib::CudaDevice>();
                                       NCudaLib::GetLatencyAndBandwidthStats<NCudaLib::CudaHost, NCudaLib::CudaDevice>();
                                   }
                                   resultPromise.set_value(TrainModelImpl(trainCatBoostOptions, outputOptions, dataProvider, testProvider, featuresManager));
                               } catch (...) {
                                   resultPromise.set_exception(std::current_exception());
                               }
                               StopCudaManager();
                           });
        thread.join();
        resultFuture.wait();

        return resultFuture.get();
    }


    void TrainModel(const NJson::TJsonValue& params,
                    const NCatboostOptions::TOutputFilesOptions& outputOptions,
                    TPool& learnPool,
                    const TPool& testPool,
                    TFullModel* model)
    {
        TString outputModelPath = outputOptions.CreateResultModelFullPath();
        NCatboostOptions::TCatBoostOptions catBoostOptions(ETaskType::GPU);
        catBoostOptions.Load(params);
        CheckForSnapshotAndReloadOptions(outputOptions, &catBoostOptions);
        SetLogingLevel(catBoostOptions.LoggingLevel);
        TDataProvider dataProvider;
        THolder<TDataProvider> testData;
        if (testPool.Docs.GetDocCount()) {
            testData = MakeHolder<TDataProvider>();
        }

        TVector<size_t> indices(learnPool.Docs.GetDocCount());
        std::iota(indices.begin(), indices.end(), 0);
        if (!catBoostOptions.DataProcessingOptions->HasTimeFlag) {
            const ui64 shuffleSeed = catBoostOptions.RandomSeed;
            TRandom random(shuffleSeed);
            Shuffle(indices.begin(), indices.end(), random);
            dataProvider.SetShuffleSeed(shuffleSeed);
        } else {
            dataProvider.SetHasTimeFlag(true);
        }
        ::ApplyPermutation(InvertPermutation(indices), &learnPool);
        auto permutationGuard = Finally([&] { ::ApplyPermutation(indices, &learnPool); });

        auto ignoredFeatures = catBoostOptions.DataProcessingOptions->IgnoredFeatures;
        const auto& systemOptions = catBoostOptions.SystemOptions.Get();
        const auto numThreads = systemOptions.NumThreads;

        TBinarizedFeaturesManager featuresManager(catBoostOptions.CatFeatureParams,
                                                  catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization);

        {
            CB_ENSURE(learnPool.Docs.GetDocCount(), "Error: empty learn pool");
            TCpuPoolBasedDataProviderBuilder builder(featuresManager, learnPool, false, dataProvider);
            builder.AddIgnoredFeatures(ignoredFeatures.Get())
                    .SetClassesWeights(catBoostOptions.DataProcessingOptions->ClassWeights)
                    .Finish(numThreads);
        }

        if (testData != nullptr) {
            TCpuPoolBasedDataProviderBuilder builder(featuresManager, testPool, true, *testData);
            builder.AddIgnoredFeatures(ignoredFeatures.Get())
                    .SetClassesWeights(catBoostOptions.DataProcessingOptions->ClassWeights)
                    .Finish(numThreads);
        }

        UpdatePinnedMemorySizeOption(dataProvider, testData.Get(), featuresManager, catBoostOptions);
        UpdateGpuSpecificDefaults(catBoostOptions, featuresManager);
        EstimatePriors(dataProvider, featuresManager, catBoostOptions.CatFeatureParams);

        auto coreModel = TrainModel(catBoostOptions, outputOptions, dataProvider, testData.Get(), featuresManager);
        auto targetClassifiers = CreateTargetClassifiers(featuresManager);
        if (model == nullptr) {
            CB_ENSURE(!outputModelPath.Size(), "Error: Model and output path are empty");
            MakeFullModel(std::move(coreModel),
                          learnPool,
                          targetClassifiers,
                          numThreads,
                          outputModelPath);
        } else
        {
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
        CheckForSnapshotAndReloadOptions(outputOptions, &catBoostOptions);
        SetLogingLevel(catBoostOptions.LoggingLevel);
        const auto resultModelPath = outputOptions.CreateResultModelFullPath();
        TString coreModelPath = TStringBuilder() << resultModelPath << ".core";

        const int numThreads = catBoostOptions.SystemOptions->NumThreads;
        TVector<TTargetClassifier> targetClassifiers;
        {
            NPar::LocalExecutor().RunAdditionalThreads(numThreads - 1);
            TBinarizedFeaturesManager featuresManager(catBoostOptions.CatFeatureParams,
                                                      catBoostOptions.DataProcessingOptions->FloatFeaturesBinarization);

            TDataProvider dataProvider;
            THolder<TDataProvider> testProvider;

            {
                MATRIXNET_INFO_LOG << "Loading data..." << Endl;

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

                NPar::TLocalExecutor localExecutor;
                localExecutor.RunAdditionalThreads(numThreads - 1);

                {

                    ReadPool(poolLoadOptions.CdFile,
                             poolLoadOptions.LearnFile,
                             poolLoadOptions.PairsFile,
                             true,
                             poolLoadOptions.Delimiter,
                             poolLoadOptions.HasHeader,
                             catBoostOptions.DataProcessingOptions->ClassNames,
                             &localExecutor,
                             &dataProviderBuilder);
                }

                if (poolLoadOptions.TestFile) {
                    MATRIXNET_INFO_LOG << "Loading test..." << Endl;
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
                             true,
                             poolLoadOptions.Delimiter,
                             poolLoadOptions.HasHeader,
                             catBoostOptions.DataProcessingOptions->ClassNames,
                             &localExecutor,
                             &testBuilder);
                }
            }

            featuresManager.UnloadCatFeaturePerfectHashFromRam();
            UpdatePinnedMemorySizeOption(dataProvider, testProvider.Get(), featuresManager, catBoostOptions);
            UpdateGpuSpecificDefaults(catBoostOptions, featuresManager);
            EstimatePriors(dataProvider, featuresManager, catBoostOptions.CatFeatureParams);


            {
                auto coreModel = TrainModel(catBoostOptions, outputOptions, dataProvider, testProvider.Get(), featuresManager);
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
