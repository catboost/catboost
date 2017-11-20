#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/helpers/eval_helpers.h>
#include "train.h"

class TGPUModelTrainer: public IModelTrainer
{
    void TrainModel(
            const NJson::TJsonValue& params,
            const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
            const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
            TPool& learnPool,
            bool allowClearPool,
            const TPool& testPool,
            const TString& outputModelPath,
            TFullModel* model,
            TEvalResult* evalResult) const override
    {
        Y_UNUSED(objectiveDescriptor);
        Y_UNUSED(evalMetricDescriptor);
        Y_UNUSED(allowClearPool);
        NCatboostCuda::TrainModel(params, learnPool, testPool, outputModelPath, model);
        evalResult->GetRawValuesRef().resize(model->ObliviousTrees.ApproxDimension);
    }
};

TTrainerFactory::TRegistrator <TGPUModelTrainer> GPURegistrator(ETaskType::GPU);

namespace NCatboostCuda
{

    inline TFullModel TrainModelImpl(const TTrainCatBoostOptions& trainCatBoostOptions,
                                     const TDataProvider& dataProvider,
                                     const TDataProvider* testProvider,
                                     TBinarizedFeaturesManager& featuresManager)
    {
        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

        if (trainCatBoostOptions.ApplicationOptions.IsProfile())
        {
            profiler.SetDefaultProfileMode(NCudaLib::EProfileMode::ImplicitLabelSync);
        } else
        {
            profiler.SetDefaultProfileMode(NCudaLib::EProfileMode::NoProfile);
        }
        TRandom random(trainCatBoostOptions.TreeConfig.GetBootstrapConfig().GetSeed());

        THolder<TAdditiveModel<TObliviousTreeModel>> model;
        const bool storeCatFeaturesInPinnedMemory = trainCatBoostOptions.BoostingOptions.UseCpuRamForCatFeaturesDataSet();

        switch (trainCatBoostOptions.TargetOptions.GetTargetType())
        {
            case ETargetFunction::RMSE:
            {
                model = Train<TL2>(featuresManager,
                                   trainCatBoostOptions.BoostingOptions,
                                   trainCatBoostOptions.OutputFilesOptions,
                                   trainCatBoostOptions.SnapshotOptions,
                                   trainCatBoostOptions.TreeConfig,
                                   trainCatBoostOptions.TargetOptions,
                                   dataProvider,
                                   testProvider,
                                   random,
                                   storeCatFeaturesInPinnedMemory);
                break;
            }
            case ETargetFunction::CrossEntropy:
            {
                model = Train<TCrossEntropy>(featuresManager,
                                             trainCatBoostOptions.BoostingOptions,
                                             trainCatBoostOptions.OutputFilesOptions,
                                             trainCatBoostOptions.SnapshotOptions,
                                             trainCatBoostOptions.TreeConfig,
                                             trainCatBoostOptions.TargetOptions,
                                             dataProvider,
                                             testProvider,
                                             random,
                                             storeCatFeaturesInPinnedMemory);
                break;
            }
            case ETargetFunction::Logloss:
            {
                model = Train<TLogloss>(featuresManager,
                                        trainCatBoostOptions.BoostingOptions,
                                        trainCatBoostOptions.OutputFilesOptions,
                                        trainCatBoostOptions.SnapshotOptions,
                                        trainCatBoostOptions.TreeConfig,
                                        trainCatBoostOptions.TargetOptions,
                                        dataProvider,
                                        testProvider,
                                        random,
                                        storeCatFeaturesInPinnedMemory);
                break;
            }

        }

        TFullModel result = ConvertToCoreModel(featuresManager,
                                               dataProvider,
                                               *model);
        {
            NJson::TJsonValue options(NJson::EJsonValueType::JSON_MAP);
            TOptionsJsonConverter<TTrainCatBoostOptions>::Save(trainCatBoostOptions, options);
            result.ModelInfo["params"] = ToString(options);
        }
        return result;
    }

    TFullModel TrainModel(const TTrainCatBoostOptions& trainCatBoostOptions,
                          const TDataProvider& dataProvider,
                          const TDataProvider* testProvider,
                          TBinarizedFeaturesManager& featuresManager)
    {
        std::promise<TFullModel> resultPromise;
        std::future<TFullModel> resultFuture = resultPromise.get_future();
        std::thread thread([&]()
                           {
                               SetLogingLevel(trainCatBoostOptions.ApplicationOptions.GetLoggingLevel());
                               NCudaLib::SetApplicationConfig(
                                       trainCatBoostOptions.ApplicationOptions.GetCudaApplicationConfig());
                               StartCudaManager(trainCatBoostOptions.ApplicationOptions.GetLoggingLevel());
                               try
                               {
                                   if (NCudaLib::GetCudaManager().GetDeviceCount() > 1)
                                   {
                                       NCudaLib::GetLatencyAndBandwidthStats<NCudaLib::CudaDevice, NCudaLib::CudaHost>();
                                       NCudaLib::GetLatencyAndBandwidthStats<NCudaLib::CudaDevice, NCudaLib::CudaDevice>();
                                       NCudaLib::GetLatencyAndBandwidthStats<NCudaLib::CudaHost, NCudaLib::CudaDevice>();
                                   }
                                   resultPromise.set_value(
                                           TrainModelImpl(trainCatBoostOptions, dataProvider, testProvider,
                                                          featuresManager));
                               } catch (...)
                               {
                                   resultPromise.set_exception(std::current_exception());
                               }
                               StopCudaManager();
                           });


        thread.join();
        resultFuture.wait();

        return resultFuture.get();
    }


    void TrainModel(const NJson::TJsonValue& params,
                    TPool& learnPool,
                    const TPool& testPool,
                    const TString& outputModelPath,
                    TFullModel* model)
    {

        TTrainCatBoostOptions catBoostOptions;
        TOptionsJsonConverter<TTrainCatBoostOptions>::Load(params, catBoostOptions);

        TDataProvider dataProvider;
        THolder<TDataProvider> testData;
        if (testPool.Docs.GetDocCount()) {
            testData = MakeHolder<TDataProvider>();
        }

        TVector<size_t> indices(learnPool.Docs.GetDocCount());
        std::iota(indices.begin(), indices.end(), 0);
        if (!catBoostOptions.BoostingOptions.HasTime()) {
            const ui64 shuffleSeed = catBoostOptions.GetShuffleSeed();
            TRandom random(shuffleSeed);
            Shuffle(indices.begin(), indices.end(), random);
            dataProvider.SetShuffleSeed(shuffleSeed);
        }
        ::ApplyPermutation(InvertPermutation(indices), &learnPool);
        auto permutationGuard = Finally([&] { ::ApplyPermutation(indices, &learnPool); });

        TBinarizedFeaturesManager featuresManager(catBoostOptions.FeatureManagerOptions);

        {
            CB_ENSURE(learnPool.Docs.GetDocCount(), "Error: empty learn pool");
            TCpuPoolBasedDataProviderBuilder builder(featuresManager, learnPool, false, dataProvider);
            builder.AddIgnoredFeatures(catBoostOptions.FeatureManagerOptions.GetIgnoredFeatures())
                    .Finish(catBoostOptions.ApplicationOptions.GetNumThreads());
        }

        if (testData != nullptr) {
            TCpuPoolBasedDataProviderBuilder builder(featuresManager, testPool, true, *testData);
            builder.AddIgnoredFeatures(catBoostOptions.FeatureManagerOptions.GetIgnoredFeatures())
                    .Finish(catBoostOptions.ApplicationOptions.GetNumThreads());
        }

        UpdatePinnedMemorySizeOption(dataProvider, testData.Get(), featuresManager, catBoostOptions);
        UpdateOptionsAndEnableCtrTypes(catBoostOptions, featuresManager);

        auto coreModel = TrainModel(catBoostOptions, dataProvider, testData.Get(), featuresManager);
        auto targetClassifiers = CreateTargetClassifiers(featuresManager);
        if (model == nullptr) {
            CB_ENSURE(!outputModelPath.Size(), "Error: Model and output path are empty");
            MakeFullModel(std::move(coreModel),
                          learnPool,
                          targetClassifiers,
                          catBoostOptions.ApplicationOptions.GetNumThreads(),
                          outputModelPath);
        } else
        {
            MakeFullModel(std::move(coreModel),
                          learnPool,
                          targetClassifiers,
                          catBoostOptions.ApplicationOptions.GetNumThreads(),
                          model);
        }
    }
}
