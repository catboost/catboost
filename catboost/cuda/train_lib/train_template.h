#pragma once

#include "train.h"

#include <catboost/cuda/methods/boosting.h>
#include <catboost/cuda/methods/oblivious_tree.h>
#include <catboost/cuda/cuda_lib/cuda_base.h>

namespace NCatboostCuda {
    template <template <class TMapping, class> class TTargetTemplate, NCudaLib::EPtrType CatFeaturesStoragePtrType>
    inline THolder<TAdditiveModel<TObliviousTreeModel>> Train(TBinarizedFeaturesManager& featureManager,
                                                              const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                              const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                              const TDataProvider& learn,
                                                              const TDataProvider* test,
                                                              TRandom& random) {
        using TTaskDataSet = TDataSet<CatFeaturesStoragePtrType>;
        using TTarget = TTargetTemplate<NCudaLib::TMirrorMapping, TTaskDataSet>;

        const bool zeroAverage = catBoostOptions.LossFunctionDescription->GetLossFunction() == ELossFunction::PairLogit;
        TObliviousTree tree(featureManager, catBoostOptions.ObliviousTreeOptions.Get(), catBoostOptions.RandomSeed, zeroAverage);
        const auto& boostingOptions = catBoostOptions.BoostingOptions.Get();
        TDynamicBoosting<TTargetTemplate, TObliviousTree, CatFeaturesStoragePtrType> boosting(featureManager,
                                                                                              boostingOptions,
                                                                                              catBoostOptions.LossFunctionDescription,
                                                                                              random,
                                                                                              tree);

        if (outputOptions.SaveSnapshot()) {
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
            if (allowWriteFiles) {
                metaOutPtr = MakeHolder<TOFStream>(outputOptions.CreateMetaFileFullPath());
            }

            if (metaOutPtr) {
                (*metaOutPtr) << "name\t" << outputOptions.GetName() << Endl;
                (*metaOutPtr) << "iterCount\t" << boostingOptions.IterationCount.Get() << Endl;
            }

            if (outputOptions.GetMetricPeriod()) {
                learnPrinter.Reset(new TMetricPrinter("learn: ", allowWriteFiles ? outputOptions.CreateLearnErrorLogFullPath() : "", "\t", "", outputOptions.GetMetricPeriod()));
                //output log files path relative to trainDirectory
                if (metaOutPtr) {
                    (*metaOutPtr) << "learnErrorLog\t" << outputOptions.CreateLearnErrorLogFullPath() << Endl;
                }
                if (test) {
                    testPrinter.Reset(
                        new TMetricPrinter("test: ", allowWriteFiles ? outputOptions.CreateTestErrorLogFullPath() : "", "\t", "\tbestTest:\t", outputOptions.GetMetricPeriod()));
                    if (metaOutPtr) {
                        (*metaOutPtr) << "testErrorLog\t" << outputOptions.CreateTestErrorLogFullPath() << Endl;
                    }

                    const auto& odOptions = boostingOptions.OverfittingDetector;
                    if (odOptions->AutoStopPValue > 0) {
                        overfitDetector = CreateOverfittingDetector(odOptions, !TTarget::IsMinOptimal(), true);
                        testPrinter->RegisterOdDetector(overfitDetector.Get());
                    }
                }
            }
            if (metaOutPtr) {
                (*metaOutPtr) << "timeLeft\t" << outputOptions.CreateTimeLeftLogFullPath() << Endl;
                TString lossDescriptionStr = ::ToString(catBoostOptions.LossFunctionDescription.Get());
                (*metaOutPtr) << "loss\t" << lossDescriptionStr << "\t"
                              << (TMetricPrinter::IsMinOptimal() ? "min" : "max")
                              << Endl;
            }
        }
        if (learnPrinter) {
            boosting.RegisterLearnListener(*learnPrinter);
        }

        if (testPrinter) {
            boosting.RegisterTestListener(*testPrinter);
        }
        if (overfitDetector) {
            boosting.AddOverfitDetector(*overfitDetector);
        }

        TTimeWriter<TTarget, TObliviousTreeModel> timeWriter(boostingOptions.IterationCount,
                                                             outputOptions.CreateTimeLeftLogFullPath(),
                                                             "\n");
        if (testPrinter) {
            boosting.RegisterTestListener(timeWriter);
        } else {
            boosting.RegisterLearnListener(timeWriter);
        }

        auto model = boosting.Run();
        if (outputOptions.ShrinkModelToBestIteration()) {
            if (testPrinter == nullptr) {
                MATRIXNET_INFO_LOG << "Warning: can't use-best-model without test set. Will skip model shrinking";
            } else {
                CB_ENSURE(testPrinter);
                const ui32 bestIter = testPrinter->GetBestIteration();
                model->Shrink(bestIter);
            }
        }
        if (testPrinter != nullptr) {
            MATRIXNET_NOTICE_LOG << "bestTest = " << testPrinter->GetBestScore() << Endl;
            MATRIXNET_NOTICE_LOG << "bestIteration = " << testPrinter->GetBestIteration() << Endl;
        }
        return model;
    }

    template <template <class TMapping, class> class TTargetTemplate>
    THolder<TAdditiveModel<TObliviousTreeModel>> Train(TBinarizedFeaturesManager& featureManager,
                                                       const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                       const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                       const TDataProvider& learn,
                                                       const TDataProvider* test,
                                                       TRandom& random,
                                                       bool storeCatFeaturesInPinnedMemory) {
        if (storeCatFeaturesInPinnedMemory) {
            return Train<TTargetTemplate, NCudaLib::EPtrType::CudaHost>(featureManager, catBoostOptions, outputOptions, learn, test, random);
        } else {
            return Train<TTargetTemplate, NCudaLib::EPtrType::CudaDevice>(featureManager, catBoostOptions, outputOptions, learn, test, random);
        }
    };

    template <template <class, class> class TTargetTemplate>
    class TGpuTrainer: public IGpuTrainer {
        virtual THolder<TAdditiveModel<TObliviousTreeModel>> TrainModel(TBinarizedFeaturesManager& featuresManager,
                                                                        const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                                        const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                        const TDataProvider& learn,
                                                                        const TDataProvider* test,
                                                                        TRandom& random,
                                                                        bool storeInPinnedMemory) const {
            return Train<TTargetTemplate>(featuresManager,
                                          catBoostOptions,
                                          outputOptions,
                                          learn,
                                          test,
                                          random,
                                          storeInPinnedMemory);
        };
    };

}
