#pragma once

#include "model_helpers.h"

#include <catboost/libs/options/catboost_options.h>
#include <catboost/cuda/data/load_data.h>
#include <catboost/cuda/methods/boosting.h>
#include <catboost/cuda/targets/mse.h>
#include <catboost/cuda/methods/oblivious_tree.h>
#include <catboost/cuda/targets/cross_entropy.h>
#include <catboost/cuda/data/cat_feature_perfect_hash.h>
#include <catboost/cuda/gpu_data/pinned_memory_estimation.h>
#include <catboost/cuda/cpu_compatibility_helpers/model_converter.h>
#include <catboost/cuda/cpu_compatibility_helpers/full_model_saver.h>
#include <catboost/cuda/cpu_compatibility_helpers/cpu_pool_based_data_provider_builder.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/train_lib/train_model.h>
#include <util/system/fs.h>

namespace NCatboostCuda
{
    inline void UpdatePinnedMemorySizeOption(const TDataProvider& learn,
                                             const TDataProvider* test,
                                             const TBinarizedFeaturesManager& featuresManager,
                                             NCatboostOptions::TCatBoostOptions& catBoostOptions)
    {
        const bool storeCatFeaturesInPinnedMemory = catBoostOptions.DataProcessingOptions->GpuCatFeaturesStorage == EGpuCatFeaturesStorage::CpuPinnedMemory;
        if (storeCatFeaturesInPinnedMemory)
        {
            ui32 devCount = NCudaLib::GetEnabledDevices(catBoostOptions.SystemOptions->Devices).size();
            ui32 cpuFeaturesSize = 104857600 + 1.05 * EstimatePinnedMemorySizeInBytesPerDevice(learn, test, featuresManager, devCount);
            ui64 currentSize = catBoostOptions.SystemOptions->PinnedMemorySize;
            if (currentSize < cpuFeaturesSize) {
                catBoostOptions.SystemOptions->PinnedMemorySize = cpuFeaturesSize;
            }
        }
    }

    inline bool HasCtrs(const TBinarizedFeaturesManager& featuresManager)
    {
        for (auto catFeature : featuresManager.GetCatFeatureIds())
        {
            if (featuresManager.UseForCtr(catFeature) || featuresManager.UseForTreeCtr(catFeature))
            {
                return true;
            }
        }
        return false;
    }

    inline void UpdateGpuSpecificDefaults(NCatboostOptions::TCatBoostOptions& options,
                                          TBinarizedFeaturesManager& featuresManager)
    {
        //don't make several permutations in matrixnet-like mode if we don't have ctrs
        if (!HasCtrs(featuresManager) && options.BoostingOptions->BoostingType == EBoostingType::Plain)
        {
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

    template<template<class TMapping, class> class TTargetTemplate, NCudaLib::EPtrType CatFeaturesStoragePtrType>
    inline THolder<TAdditiveModel<TObliviousTreeModel>> Train(TBinarizedFeaturesManager& featureManager,
                                                              const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                              const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                              const TDataProvider& learn,
                                                              const TDataProvider* test,
                                                              TRandom& random)
    {
        using TTaskDataSet = TDataSet<CatFeaturesStoragePtrType>;
        using TTarget = TTargetTemplate<NCudaLib::TMirrorMapping, TTaskDataSet>;

        TObliviousTree tree(featureManager, catBoostOptions.ObliviousTreeOptions.Get());
        const auto& boostingOptions = catBoostOptions.BoostingOptions.Get();
        TDynamicBoosting<TTargetTemplate, TObliviousTree, CatFeaturesStoragePtrType> boosting(featureManager,
                                                                                              boostingOptions,
                                                                                              catBoostOptions.LossFunctionDescription,
                                                                                              random,
                                                                                              tree);

        if (outputOptions.SaveSnapshot())
        {
            NJson::TJsonValue options;
            catBoostOptions.Save(&options);
            auto optionsStr = ToString<NJson::TJsonValue>(options);
            boosting.SaveSnapshot(outputOptions.CreateSnapshotFullPath(), optionsStr, outputOptions.GetSnapshotSaveInterval());
        }
        boosting.SetDataProvider(learn, test);

        using TMetricPrinter = TMetricLogger<TTarget, TObliviousTreeModel>;
        TIterationLogger<TTarget, TObliviousTreeModel> iterationPrinter(":\t");

        THolder<IOverfittingDetector> overfitDetector;
        boosting.RegisterLearnListener(iterationPrinter);

        THolder<TMetricPrinter> learnPrinter;
        THolder<TMetricPrinter> testPrinter;

        {
            THolder<TOFStream> metaOutPtr;
            const bool allowWriteFiles = outputOptions.AllowWriteFiles();
            if (allowWriteFiles)
            {
                metaOutPtr = MakeHolder<TOFStream>(outputOptions.CreateMetaFileFullPath());
            }

            if (metaOutPtr) {
                (*metaOutPtr) << "name\t" << outputOptions.GetName() << Endl;
                (*metaOutPtr) << "iterCount\t" << boostingOptions.IterationCount.Get() << Endl;
            }

            if (outputOptions.GetMetricPeriod())
            {
                learnPrinter.Reset(new TMetricPrinter("learn: ", allowWriteFiles ? outputOptions.CreateLearnErrorLogFullPath() : "", "\t", "", outputOptions.GetMetricPeriod()));
                //output log files path relative to trainDirectory
                if (metaOutPtr) {
                    (*metaOutPtr) << "learnErrorLog\t" << outputOptions.CreateLearnErrorLogFullPath() << Endl;
                }
                if (test)
                {
                    testPrinter.Reset(
                            new TMetricPrinter("test: ", allowWriteFiles ? outputOptions.CreateTestErrorLogFullPath() : "", "\t", "\tbestTest:\t", outputOptions.GetMetricPeriod()));
                    if (metaOutPtr)
                    {
                        (*metaOutPtr) << "testErrorLog\t" << outputOptions.CreateTestErrorLogFullPath() << Endl;
                    }

                    const auto& odOptions = boostingOptions.OverfittingDetector;
                    if (odOptions->AutoStopPValue > 0)
                    {
                        overfitDetector = CreateOverfittingDetector(odOptions, !TTarget::IsMinOptimal(), true);
                        testPrinter->RegisterOdDetector(overfitDetector.Get());
                    }
                }
            }
            if (metaOutPtr)
            {
                (*metaOutPtr) << "timeLeft\t" << outputOptions.CreateTimeLeftLogFullPath() << Endl;
                (*metaOutPtr) << "loss\t" << TMetricPrinter::GetMetricName() << "\t"
                              << (TMetricPrinter::IsMinOptimal() ? "min" : "max")
                              << Endl;
            }
        }
        if (learnPrinter)
        {
            boosting.RegisterLearnListener(*learnPrinter);
        }

        if (testPrinter)
        {
            boosting.RegisterTestListener(*testPrinter);
        }
        if (overfitDetector)
        {
            boosting.AddOverfitDetector(*overfitDetector);
        }

        TTimeWriter<TTarget, TObliviousTreeModel> timeWriter(boostingOptions.IterationCount,
                                                             outputOptions.CreateTimeLeftLogFullPath(),
                                                             "\n");
        if (testPrinter)
        {
            boosting.RegisterTestListener(timeWriter);
        } else
        {
            boosting.RegisterLearnListener(timeWriter);
        }

        auto model = boosting.Run();
        if (outputOptions.ShrinkModelToBestIteration())
        {
            if (testPrinter == nullptr)
            {
                MATRIXNET_INFO_LOG << "Warning: can't use-best-model without test set. Will skip model shrinking";
            } else
            {
                CB_ENSURE(testPrinter);
                const ui32 bestIter = testPrinter->GetBestIteration();
                model->Shrink(bestIter);
            }
        }
        if (testPrinter != nullptr)
        {
            MATRIXNET_NOTICE_LOG << "bestTest = " << testPrinter->GetBestScore() << Endl;
            MATRIXNET_NOTICE_LOG << "bestIteration = " << testPrinter->GetBestIteration() << Endl;
        }
        return model;
    }

    template<template<class TMapping, class> class TTargetTemplate>
    inline THolder<TAdditiveModel<TObliviousTreeModel>> Train(TBinarizedFeaturesManager& featureManager,
                                                              const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                              const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                              const TDataProvider& learn,
                                                              const TDataProvider* test,
                                                              TRandom& random,
                                                              bool storeCatFeaturesInPinnedMemory) {
        if (storeCatFeaturesInPinnedMemory)
        {
            return Train<TTargetTemplate, NCudaLib::CudaHost>(featureManager, catBoostOptions, outputOptions, learn, test, random);
        } else
        {
            return Train<TTargetTemplate, NCudaLib::CudaDevice>(featureManager, catBoostOptions, outputOptions, learn, test, random);
        }
    };


    TFullModel TrainModel(const NCatboostOptions::TCatBoostOptions& trainCatBoostOptions,
                          const NCatboostOptions::TOutputFilesOptions& outputOptions,
                          const TDataProvider& dataProvider,
                          const TDataProvider* testProvider,
                          TBinarizedFeaturesManager& featuresManager);


    void TrainModel(const NJson::TJsonValue& params,
                    const NCatboostOptions::TOutputFilesOptions& outputOptions,
                    TPool& learnPool,
                    const TPool& testPool,
                    TFullModel* model);


    void TrainModel(const NCatboostOptions::TPoolLoadParams& poolLoadOptions,
                    const NCatboostOptions::TOutputFilesOptions& outputOptions,
                    const NJson::TJsonValue& jsonOptions);

}
