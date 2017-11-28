#include "train.h"
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/helpers/eval_helpers.h>

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
        if (outputOptions.SaveSnapshot() && NFs::Exists(outputOptions.CreateSnapshotFullPath())) {
            TString jsonOptions;
            TProgressHelper(ToString<ETaskType>(options->GetTaskType())).CheckedLoad(
                    outputOptions.CreateSnapshotFullPath(), [&](TIFStream* in) {
                ::Load(in, jsonOptions);
            });
            options->Load(jsonOptions);
        }
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
