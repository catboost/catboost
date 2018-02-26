#pragma once

#include "train.h"

#include <catboost/cuda/methods/dynamic_boosting.h>
#include <catboost/cuda/methods/feature_parallel_pointwise_oblivious_tree.h>
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/methods/doc_parallel_pointwise_oblivious_tree.h>
#include <catboost/cuda/methods/doc_parallel_boosting.h>

namespace NCatboostCuda {
    template <class TBoosting>
    inline THolder<TAdditiveModel<typename TBoosting::TWeakModel>> Train(TBinarizedFeaturesManager& featureManager,
                                                                         const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                                         const NCatboostOptions::TOutputFilesOptions& outputOptions,
                                                                         const TDataProvider& learn,
                                                                         const TDataProvider* test,
                                                                         TRandom& random) {
        using TWeakLearner = typename TBoosting::TWeakLearner;
        using TWeakModel = typename TBoosting::TWeakModel;
        using TObjective = typename TBoosting::TObjective;

        const bool zeroAverage = catBoostOptions.LossFunctionDescription->GetLossFunction() == ELossFunction::PairLogit;
        TWeakLearner weak(featureManager,
                          catBoostOptions,
                          zeroAverage);

        const auto& boostingOptions = catBoostOptions.BoostingOptions.Get();
        TBoosting boosting(featureManager,
                           boostingOptions,
                           catBoostOptions.LossFunctionDescription,
                           random,
                           weak);

        if (outputOptions.SaveSnapshot()) {
            NJson::TJsonValue options;
            catBoostOptions.Save(&options);
            auto optionsStr = ToString<NJson::TJsonValue>(options);
            boosting.SaveSnapshot(outputOptions.CreateSnapshotFullPath(), optionsStr, outputOptions.GetSnapshotSaveInterval());
        }
        boosting.SetDataProvider(learn,
                                 test);

        using TMetricPrinter = TMetricLogger<TObjective, TWeakModel>;
        TIterationLogger<TObjective, TWeakModel> iterationPrinter(":\t");

        THolder<IOverfittingDetector> overfitDetector;
        boosting.RegisterLearnListener(iterationPrinter);

        THolder<TMetricPrinter> learnPrinter;
        THolder<TMetricPrinter> testPrinter;

        //TODO(noxoomo): to new CPU logger
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
                        overfitDetector = CreateOverfittingDetector(odOptions, !TObjective::IsMinOptimal(), true);
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

        TTimeWriter<TObjective, TObliviousTreeModel> timeWriter(boostingOptions.IterationCount,
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
        if (catBoostOptions.BoostingOptions->DataPartitionType == EDataPartitionType::FeatureParallel) {
            using TFeatureParallelWeakLearner = TFeatureParallelPointwiseObliviousTree;
#define TRAIN_FEATURE_PARALLEL(PtrType)                                                        \
    using TBoosting = TDynamicBoosting<TTargetTemplate, TFeatureParallelWeakLearner, PtrType>; \
    return Train<TBoosting>(featureManager, catBoostOptions, outputOptions, learn, test, random);

            if (storeCatFeaturesInPinnedMemory) {
                TRAIN_FEATURE_PARALLEL(NCudaLib::EPtrType::CudaHost)
            } else {
                TRAIN_FEATURE_PARALLEL(NCudaLib::EPtrType::CudaDevice)
            }
#undef TRAIN_FEATURE_PARALLEL

        } else {
            using TDocParallelBoosting = TBoosting<TTargetTemplate, TDocParallelObliviousTree>;
            return Train<TDocParallelBoosting>(featureManager, catBoostOptions, outputOptions,
                                               learn, test, random);
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
