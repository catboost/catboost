#pragma once

#include "learning_rate.h"
#include "random_score_helper.h"
#include "doc_parallel_boosting_progress.h"
#include "boosting_progress_tracker.h"

#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset_builder.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/targets/target_func.h>

#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/metrics/optimal_const_for_loss.h>
#include <catboost/private/libs/options/boosting_options.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/libs/overfitting_detector/overfitting_detector.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/stream/format.h>

namespace NCatboostCuda {
    template <template <class TMapping> class TTargetTemplate,
              class TWeakLearner_>
    class TBoosting {
    public:
        using TObjective = TTargetTemplate<NCudaLib::TStripeMapping>;
        using TWeakLearner = TWeakLearner_;
        using TResultModel = TAdditiveModel<typename TWeakLearner::TResultModel>;
        using TWeakModel = typename TWeakLearner::TResultModel;
        using TVec = typename TObjective::TVec;
        using TConstVec = typename TObjective::TConstVec;
        using TDataSet = TDocParallelDataSet;
        using TCursor = TStripeBuffer<float>;

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const NCB::TTrainingDataProvider* DataProvider = nullptr;
        const NCB::TTrainingDataProvider* TestDataProvider = nullptr;
        const NCB::TFeatureEstimators* Estimators = nullptr;
        TBoostingProgressTracker* ProgressTracker = nullptr;

        TGpuAwareRandom& Random;
        ui64 BaseIterationSeed;
        const NCatboostOptions::TCatBoostOptions& CatBoostOptions;
        const NCatboostOptions::TBoostingOptions& Config;
        const NCatboostOptions::TModelBasedEvalOptions& ModelBasedEvalConfig;
        const NCatboostOptions::TLossDescription& TargetOptions;
        const TMaybe<TCustomObjectiveDescriptor> ObjectiveDescriptor;

        NPar::ILocalExecutor* LocalExecutor;

    private:
        inline static TDocParallelDataSetsHolder CreateDocParallelDataSet(TBinarizedFeaturesManager& manager,
                                                                          const NCB::TTrainingDataProvider& dataProvider,
                                                                          const NCB::TFeatureEstimators& estimators,
                                                                          const NCB::TTrainingDataProvider* test,
                                                                          ui32 permutationCount,
                                                                          NPar::ILocalExecutor* localExecutor) {
            TDocParallelDataSetBuilder dataSetsHolderBuilder(manager,
                                                             dataProvider,
                                                             estimators,
                                                             test
                                                             );
            return dataSetsHolderBuilder.BuildDataSet(permutationCount, localExecutor);
        }

        struct TBoostingCursors {
            using TCursor = TStripeBuffer<float>;
            TVector<TCursor> Cursors; // [permutationIdx]
            TVec TestCursor;
            THolder<TVec> BestTestCursor;
            TMaybe<TVector<double>> StartingPoint;

            void CopyFrom(const TBoostingCursors& other) {
                Cursors.resize(other.Cursors.size());
                for (ui32 idx : xrange(other.Cursors.size())) {
                    Cursors[idx] = TStripeBuffer<float>::CopyMappingAndColumnCount(other.Cursors[idx]);
                    Cursors[idx].Copy(other.Cursors[idx]);
                }
                TestCursor = TStripeBuffer<float>::CopyMappingAndColumnCount(other.TestCursor);
                TestCursor.Copy(other.TestCursor);
                if (other.BestTestCursor != nullptr) {
                    BestTestCursor = MakeHolder<TStripeBuffer<float>>();
                    *BestTestCursor = TStripeBuffer<float>::CopyMappingAndColumnCount(*other.BestTestCursor);
                    BestTestCursor->Copy(*other.BestTestCursor);
                }
            }
        };

        struct TBoostingInputData {
            TDocParallelDataSetsHolder DataSets;

            TVector<THolder<TObjective>> Targets;
            THolder<TObjective> TestTarget;

            ui32 GetEstimationPermutation() const {
                return DataSets.PermutationsCount() - 1;
            }
        };

        THolder<TBoostingInputData> CreateInputData(ui32 permutationCount, TBinarizedFeaturesManager* featureManager) {
            CB_ENSURE(DataProvider);
            const auto& dataProvider = *DataProvider;
            THolder<TBoostingInputData> inputData(new TBoostingInputData);

            inputData->DataSets = CreateDocParallelDataSet(*featureManager,
                                                           dataProvider,
                                                           *Estimators,
                                                           TestDataProvider,
                                                           permutationCount,
                                                           LocalExecutor);

            for (ui32 i = 0; i < permutationCount; ++i) {
                inputData->Targets.push_back(CreateTarget(inputData->DataSets.GetDataSetForPermutation(i)));
            }
            if (TestDataProvider) {
                inputData->TestTarget = CreateTarget(inputData->DataSets.GetTestDataSet());
            }

            const ui32 approxDim = inputData->Targets[0]->GetDim();
            for (ui32 i = 1; i < permutationCount; ++i) {
                CB_ENSURE(approxDim == inputData->Targets[i]->GetDim(), "Approx dim should be consistent. This is a bug: report to catboost team");
            }
            if (inputData->TestTarget) {
                CB_ENSURE(approxDim == inputData->TestTarget->GetDim(),
                          "Approx dim should be consistent. This is a bug: report to catboost team");
            }

            return inputData;
        }

        THolder<TBoostingCursors> CreateCursors(const TBoostingInputData& inputData) {
            CB_ENSURE(DataProvider);
            const auto& dataProvider = *DataProvider;
            THolder<TBoostingCursors> cursors(new TBoostingCursors);

            cursors->Cursors.resize(inputData.DataSets.PermutationsCount());
            const ui32 permutationCount = inputData.Targets.size();
            const ui32 approxDim = inputData.Targets[0]->GetDim();

            const bool isBoostFromAverage = CatBoostOptions.BoostingOptions->BoostFromAverage.Get();
            const bool isRMSEWithUncertainty = CatBoostOptions.LossFunctionDescription->GetLossFunction() == ELossFunction::RMSEWithUncertainty;
            if (isBoostFromAverage || isRMSEWithUncertainty) {
                CB_ENSURE(
                    !DataProvider->TargetData->GetBaseline()
                    && (!TestDataProvider || !TestDataProvider->TargetData->GetBaseline()),
                    "You can't use boost_from_average or RMSEWithUncertainty with baseline now.");
                cursors->StartingPoint = NCB::CalcOptimumConstApprox(
                    CatBoostOptions.LossFunctionDescription,
                    *DataProvider->TargetData->GetTarget(),
                    GetWeights(*DataProvider->TargetData));
            }

            for (ui32 i = 0; i < permutationCount; ++i) {
                const auto& loadBalancingPermutation = inputData.DataSets.GetLoadBalancingPermutation();
                cursors->Cursors[i].Reset(inputData.DataSets.GetDataSetForPermutation(0).GetTarget().GetSamplesMapping(), approxDim);
                CB_ENSURE(cursors->Cursors[i].GetMapping().GetObjectsSlice().Size());

                if (dataProvider.MetaInfo.BaselineCount > 0) {
                    auto dataProviderBaseline = *dataProvider.TargetData->GetBaseline();

                    TVector<float> baselineBias;
                    if (dataProvider.MetaInfo.BaselineCount > approxDim) {
                        CB_ENSURE(approxDim + 1 == dataProvider.MetaInfo.BaselineCount);
                        baselineBias = loadBalancingPermutation.Gather(dataProviderBaseline[approxDim]);
                    }
                    for (ui32 dim = 0; dim < approxDim; ++dim) {
                        TVector<float> baseline = loadBalancingPermutation.Gather(dataProviderBaseline[dim]);
                        for (ui32 j = 0; j < baselineBias.size(); ++j) {
                            baseline[j] -= baselineBias[j];
                        }
                        CB_ENSURE(baseline.size() == cursors->Cursors[i].GetObjectsSlice().Size());
                        cursors->Cursors[i].ColumnView(dim).Write(baseline);
                    }
                } else {
                    const auto sampleCount = DataProvider->GetObjectCount();
                    for (ui32 dim = 0; dim < approxDim; ++dim) {
                        const float value = cursors->StartingPoint ? (*cursors->StartingPoint)[dim] : 0.0;
                        const TVector<float> start(sampleCount, value);
                        cursors->Cursors[i].ColumnView(dim).Write(start);
                    }
                }
            }

            if (TestDataProvider) {
                cursors->TestCursor.Reset(inputData.DataSets.GetTestDataSet().GetTarget().GetSamplesMapping(), approxDim);
                if (TestDataProvider->MetaInfo.BaselineCount > 0) {
                    auto testDataProviderBaseline = *TestDataProvider->TargetData->GetBaseline();

                    const auto& testPermutation = inputData.DataSets.GetTestLoadBalancingPermutation();
                    TVector<float> baselineBias;
                    if (TestDataProvider->MetaInfo.BaselineCount > approxDim) {
                        CB_ENSURE(approxDim + 1 == TestDataProvider->MetaInfo.BaselineCount);
                        baselineBias = testPermutation.Gather(testDataProviderBaseline[approxDim]);
                    }
                    for (ui32 dim = 0; dim < approxDim; ++dim) {
                        TVector<float> baseline = testPermutation.Gather(testDataProviderBaseline[dim]);
                        for (ui32 i = 0; i < baselineBias.size(); ++i) {
                            baseline[i] -= baselineBias[i];
                        }
                        cursors->TestCursor.ColumnView(dim).Write(baseline);
                    }
                } else {
                    const auto sampleCount = TestDataProvider->GetObjectCount();
                    for (ui32 dim = 0; dim < approxDim; ++dim) {
                        const float value = cursors->StartingPoint ? (*cursors->StartingPoint)[dim] : 0.0;
                        const TVector<float> start(sampleCount, value);
                        cursors->TestCursor.ColumnView(dim).Write(start);
                    }
                }
            }

            if (ProgressTracker->NeedBestTestCursor()) {
                CB_ENSURE(TestDataProvider, "Need test data provider");
                cursors->BestTestCursor = MakeHolder<TStripeBuffer<float>>();
                (*cursors->BestTestCursor) = TStripeBuffer<float>::CopyMappingAndColumnCount(cursors->TestCursor);
            }
            return cursors;
        }

        void MaybeRestoreBestTestCursorAndModelsFromSnapshot(
            const TBoostingInputData& inputData,
            TBoostingProgressTracker* progressTracker,
            TVec* bestTestCursor,
            TVector<TResultModel>* models
        ) {
            const ui32 permutationCount = inputData.Targets.size();
            models->resize(permutationCount);
            progressTracker->MaybeRestoreFromSnapshot([&](IInputStream* in) {
                using TProgress = TBoostingProgress<TResultModel>;
                TProgress progress;
                ::Load(in, progress);
                if (bestTestCursor) {
                    LoadCudaBuffer(in, bestTestCursor);
                }
                *models = RestoreFromProgress(FeaturesManager, progress);
            });
        }

        void AppendEnsembles(const TDocParallelDataSetsHolder& dataSets,
                             const TVector<TResultModel>& ensembles,
                             ui32 estimationPermutation,
                             ui32 iterStart,
                             ui32 iterEnd,
                             TWeakLearner* weak,
                             TVector<TVec>* cursors,
                             TVec* testCursor) {
            TVector<TWeakModel> iterationWeakModels;
            iterationWeakModels.resize(ensembles.size());
            CB_ENSURE(iterEnd <= ensembles[0].Size(),
                "End iteration " + ToString(iterEnd) + " is outside ensemble " + ToString(ensembles[0].Size()));
            for (ui32 iter = iterStart; iter < iterEnd; ++iter) {
                for (ui32 permutation = 0; permutation < ensembles.size(); ++permutation) {
                    iterationWeakModels[permutation] = ensembles[permutation][iter];
                }
                AppendModels(dataSets, iterationWeakModels, estimationPermutation, weak, cursors, testCursor);
            }
        }

        void AppendModels(const TDocParallelDataSetsHolder& dataSets,
                          const TVector<TWeakModel>& iterationsModels,
                          ui32 estimationPermutation,
                          TWeakLearner* weak,
                          TVector<TCursor>* cursors,
                          TCursor* testCursor) {
            const bool streamParallelAppend = false;
            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();
            auto appendModelTime = profiler.Profile("Append models time");
            auto addModelValue = weak->template CreateAddModelValue<TDocParallelDataSet>(streamParallelAppend);

            if (dataSets.HasTestDataSet()) {
                addModelValue.AddTask(iterationsModels[estimationPermutation],
                                      dataSets.GetTestDataSet(),
                                      *testCursor);
            }

            for (ui32 permutation = 0; permutation < dataSets.PermutationsCount(); ++permutation) {
                const auto& ds = dataSets.GetDataSetForPermutation(permutation);
                addModelValue.AddTask(iterationsModels[permutation],
                                      ds,
                                      (*cursors)[permutation]);
            }

            addModelValue.Proceed();
        }

        THolder<TObjective> CreateTarget(const TDocParallelDataSet& dataSet) const {
            return MakeHolder<TObjective>(
                dataSet,
                Random,
                TargetOptions,
                ObjectiveDescriptor);
        }

        //TODO(noxoomo): remove overhead of multiple target for permutation datasets
        //don't look ahead boosting
        void Fit(const TDocParallelDataSetsHolder& dataSet,
                 const ui32 estimationPermutation,
                 const TVector<const TObjective*>& learnTarget,
                 const TObjective* testTarget,
                 TWeakLearner* weak,
                 TBoostingProgressTracker* progressTracker,
                 TVector<TCursor>* learnCursors,
                 TVec* testCursor,
                 TVector<TResultModel>* result,
                 TVec* bestTestCursor) {

            auto& profiler = NCudaLib::GetProfiler();

            const ui32 permutationCount = dataSet.PermutationsCount();
            CB_ENSURE(permutationCount >= 1);
            const ui32 learnPermutationCount = estimationPermutation ? permutationCount - 1 : 1; //fallback
            CB_ENSURE(learnCursors->size() == permutationCount);

            const double step = Config.LearningRate;

            auto startTimeBoosting = Now();
            TMetricCalcer<TObjective> learnMetricCalcer(*learnTarget[estimationPermutation], LocalExecutor);
            THolder<TMetricCalcer<TObjective>> testMetricCalcer;
            if (testTarget) {
                testMetricCalcer = MakeHolder<TMetricCalcer<TObjective>>(*testTarget, LocalExecutor);
            }

            auto snapshotSaver = [&](IOutputStream* out) {
                auto progress = MakeProgress(FeaturesManager, *result);
                ::Save(out, progress);
                if (bestTestCursor) {
                    SaveCudaBuffer(*bestTestCursor, out);
                }
            };

            while (!(progressTracker->ShouldStop())) {
                CheckInterrupted(); // check after long-lasting operation
                auto iterationTimeGuard = profiler.Profile("Boosting iteration");
                progressTracker->MaybeSaveSnapshot(snapshotSaver);
                const auto iteration = progressTracker->GetCurrentIteration();
                TRandom rand(iteration + BaseIterationSeed);
                rand.Advance(10);
                TOneIterationProgressTracker iterationProgressTracker(*progressTracker);

                TVector<TWeakModel> iterationModels = [&]() -> TVector<TWeakModel> {
                    auto guard = profiler.Profile("Search for weak model structure");
                    const ui32 learnPermutationId = learnPermutationCount > 1 ? static_cast<const ui32>(rand.NextUniformL() %
                                                                                                        (learnPermutationCount - 1))
                                                                              : 0;

                    const auto& taskDataSet = dataSet.GetDataSetForPermutation(learnPermutationId);
                    using TWeakTarget = typename TTargetAtPointTrait<TObjective>::Type;
                    auto target = TTargetAtPointTrait<TObjective>::Create(
                        *(learnTarget[learnPermutationId]),
                        (*learnCursors)[learnPermutationId].AsConstBuf()
                    );
                    auto mult = CalcScoreModelLengthMult(dataSet.GetDataProvider().GetObjectCount(),
                                                         iteration * step);
                    auto optimizer = weak->template CreateStructureSearcher<TWeakTarget, TDocParallelDataSet>(
                        mult,
                        (*result)[learnPermutationId]);
                    //search for best model and values of shifted target
                    auto model = optimizer.Fit(taskDataSet,
                                               target);
                    TVector<TWeakModel> models;
                    models.resize(result->size(), model);
                    return models;
                }();

                if (weak->NeedEstimation()) {
                    auto estimateModelsGuard = profiler.Profile("Estimate models");
                    auto estimator = weak->CreateEstimator();

                    for (ui32 permutation = 0; permutation < permutationCount; ++permutation) {
                        const auto& taskDataSet = dataSet.GetDataSetForPermutation(permutation);
                        estimator.AddEstimationTask(
                            *(learnTarget[permutation]),
                            taskDataSet,
                            (*learnCursors)[permutation].AsConstBuf(),
                            &iterationModels[permutation]
                        );
                    }
                    estimator.Estimate(LocalExecutor);
                }
                //
                for (auto& iterationModel : iterationModels) {
                    iterationModel.Rescale(step);
                };

                AppendModels(dataSet,
                             iterationModels,
                             estimationPermutation,
                             weak,
                             learnCursors,
                             testCursor);

                for (ui32 i = 0; i < iterationModels.size(); ++i) {
                    (*result)[i].AddWeakModel(iterationModels[i]);
                }

                {
                    auto learnListenerTimeGuard = profiler.Profile("Boosting listeners time: Learn");
                    learnMetricCalcer.SetPoint((*learnCursors)[estimationPermutation].ConstCopyView());
                    iterationProgressTracker.TrackLearnErrors(learnMetricCalcer);
                }

                if (dataSet.HasTestDataSet()) {
                    auto testListenerTimeGuard = profiler.Profile("Boosting listeners time: Test");

                    testMetricCalcer->SetPoint(testCursor->ConstCopyView());
                    iterationProgressTracker.TrackTestErrors(*testMetricCalcer);
                }

                if (bestTestCursor && iterationProgressTracker.IsBestTestIteration()) {
                    CB_ENSURE(testCursor, "Need cursor for test data");
                    bestTestCursor->Copy(*testCursor);
                }
            }

            progressTracker->MaybeSaveSnapshot(snapshotSaver);
            auto gatherCpuApproxByCursor = [&](TVec* cursor) -> TVector<TVector<double>> {
                TVector<TVector<double>> cpuApproxPermuted;
                ReadApproxInCpuFormat(*cursor, TargetOptions.GetLossFunction() == ELossFunction::MultiClass, &cpuApproxPermuted);
                TVector<ui32> order;
                dataSet.GetTestLoadBalancingPermutation().FillOrder(order);
                TVector<TVector<double>> cpuApprox(cpuApproxPermuted.size());
                for (ui64 i = 0; i < cpuApproxPermuted.size(); ++i) {
                    cpuApprox[i] = Scatter(order, cpuApproxPermuted[i]);
                }
                return cpuApprox;
            };
            if (bestTestCursor) {
                progressTracker->SetBestTestCursor(gatherCpuApproxByCursor(bestTestCursor));
            }
            if (dataSet.HasTestDataSet()) {
                progressTracker->SetFinalTestCursor(gatherCpuApproxByCursor(testCursor));
            }
            CATBOOST_INFO_LOG << "Total time " << (Now() - startTimeBoosting).SecondsFloat() << Endl;
        }

        using TEnsemble = TAdditiveModel<TWeakModel>;

    public:
        TBoosting(TBinarizedFeaturesManager& binarizedFeaturesManager,
                  const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                  const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                  EGpuCatFeaturesStorage,
                  TGpuAwareRandom& random,
                  NPar::ILocalExecutor* localExecutor)
            : FeaturesManager(binarizedFeaturesManager)
            , Random(random)
            , BaseIterationSeed(random.NextUniformL())
            , CatBoostOptions(catBoostOptions)
            , Config(catBoostOptions.BoostingOptions)
            , ModelBasedEvalConfig(catBoostOptions.ModelBasedEvalOptions)
            , TargetOptions(catBoostOptions.LossFunctionDescription)
            , ObjectiveDescriptor(objectiveDescriptor)
            , LocalExecutor(localExecutor)
        {
        }

        virtual ~TBoosting() = default;

        //TODO(noxoomo): to common with dynamic boosting superclass
        TBoosting& SetDataProvider(const NCB::TTrainingDataProvider& learnData,
                                   const NCB::TFeatureEstimators& estimators,
                                   const NCB::TTrainingDataProvider* testData = nullptr
        ) {
            DataProvider = &learnData;
            Estimators = &estimators;
            TestDataProvider = testData;
            return *this;
        }

        void SetBoostingProgressTracker(TBoostingProgressTracker* progressTracker) {
            ProgressTracker = progressTracker;
        }

        THolder<TResultModel> Run() {
            CB_ENSURE(DataProvider, "Error: set dataProvider first");
            CB_ENSURE(ProgressTracker, "Error: set boosting tracker first");

            const ui32 permutationCount = Config.PermutationCount;
            auto inputData = CreateInputData(permutationCount, &FeaturesManager);
            auto cursors = CreateCursors(*inputData);

            TVector<TEnsemble> models;
            MaybeRestoreBestTestCursorAndModelsFromSnapshot(
                *inputData,
                ProgressTracker,
                cursors->BestTestCursor.Get(),
                &models
            );
            CB_ENSURE(
                models.size() == permutationCount,
                "Progress permutation count differs from current learning task: " << models.size() << " / " << permutationCount
            );
            auto weak = MakeWeakLearner<TWeakLearner>(FeaturesManager, Config, CatBoostOptions, Random);
            if (models[0].Size() > 0) {
                auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile("Restore from progress");
                AppendEnsembles(
                    inputData->DataSets,
                    models,
                    inputData->GetEstimationPermutation(),
                    /*iterStart*/ 0,
                    /*iterEnd*/ models[0].Size(),
                    &weak,
                    &cursors->Cursors,
                    TestDataProvider ? &cursors->TestCursor : nullptr
                );
                CATBOOST_DEBUG_LOG << "Restore #" << models[0].Size() << " trees from progress" << Endl;
            }

            Fit(inputData->DataSets,
                inputData->GetEstimationPermutation(),
                GetConstPointers(inputData->Targets),
                inputData->TestTarget.Get(),
                &weak,
                ProgressTracker,
                &(cursors->Cursors),
                TestDataProvider ? &(cursors->TestCursor) : nullptr,
                &models,
                cursors->BestTestCursor.Get()
            );
            auto& modelToExport = models[inputData->GetEstimationPermutation()];
            modelToExport.SetBias(cursors->StartingPoint);
            return MakeHolder<TResultModel>(modelToExport);
        }

        void RunModelBasedEval() {
            CB_ENSURE(DataProvider && TestDataProvider, "Error: set learn and test data providers first");
            CB_ENSURE(ProgressTracker, "Error: set boosting tracker first");

            const ui32 permutationCount = Config.PermutationCount;
            const auto& features = ModelBasedEvalConfig.FeaturesToEvaluate.Get();
            TSet<ui32> allEvaluatedFeatures;
            for (const auto& featureSet : features) {
                allEvaluatedFeatures.insert(featureSet.begin(), featureSet.end());
            }
            const auto isFullBaseline = ModelBasedEvalConfig.UseEvaluatedFeaturesInBaselineModel;
            const auto& ignoredFeaturesInBaseline = isFullBaseline ? TSet<ui32>() : allEvaluatedFeatures;
            TBinarizedFeaturesManager baseFeatureManager(
                FeaturesManager,
                /*ignoredFeatureIds*/{ignoredFeaturesInBaseline.begin(), ignoredFeaturesInBaseline.end()}
            );
            auto baseInputData = CreateInputData(permutationCount, &baseFeatureManager);
            auto baseCursors = CreateCursors(*baseInputData);

            auto forceBaseSnapshotLoadFunc = [&] (
                    NCatboostOptions::TCatBoostOptions* catboostOptions,
                    NCatboostOptions::TOutputFilesOptions* outputOptions
            ) {
                outputOptions->SetSaveSnapshotFlag(true); // same call for save and load
                outputOptions->SetSnapshotFilename(catboostOptions->ModelBasedEvalOptions->BaselineModelSnapshot.Get());
            };
            auto baseSnapshotLoader = ProgressTracker->Clone(forceBaseSnapshotLoadFunc);
            TVector<TEnsemble> baseModels;
            MaybeRestoreBestTestCursorAndModelsFromSnapshot(
                *baseInputData,
                baseSnapshotLoader.Get(),
                baseCursors->BestTestCursor.Get(),
                &baseModels
            );
            for (auto& model : baseModels) {
                baseSnapshotLoader->ShrinkToBestIteration(&model);
            }

            const ui32 baseModelSize = baseModels[0].Size();
            const ui32 offset = ModelBasedEvalConfig.Offset;
            CB_ENSURE(baseModelSize >= offset, "Error: offset " << offset << " must be less or equal to best iteration of baseline model " << baseModelSize);
            const ui32 experimentCount = ModelBasedEvalConfig.ExperimentCount;
            const auto getExperimentStart = [=] (ui32 experimentIdx) {
                return baseModelSize - offset + offset / experimentCount * experimentIdx;
            };

            auto baseWeak = MakeWeakLearner<TWeakLearner>(baseFeatureManager, Config, CatBoostOptions, Random);
            AppendEnsembles(
                baseInputData->DataSets,
                baseModels,
                baseInputData->GetEstimationPermutation(),
                /*iterStart*/ 0,
                /*iterEnd*/ getExperimentStart(/*experimentIdx*/ 0),
                &baseWeak,
                &baseCursors->Cursors,
                TestDataProvider ? &baseCursors->TestCursor : nullptr
            );
            auto startingBaseCursors = CreateCursors(*baseInputData);
            startingBaseCursors->CopyFrom(*baseCursors);

            const ui32 experimentSize = ModelBasedEvalConfig.ExperimentSize;
            const ui64 savedBaseSeed = BaseIterationSeed;
            size_t featureSetIdx;
            int experimentIdx;
            auto forceMetricSaveFunc = [&] (
                NCatboostOptions::TCatBoostOptions* catboostOptions,
                NCatboostOptions::TOutputFilesOptions* outputOptions
            ) {
                catboostOptions->BoostingOptions->IterationCount.Set(experimentSize);
                outputOptions->SetSaveSnapshotFlag(false);
                outputOptions->SetMetricPeriod(1);
                const auto trainDir = JoinFsPaths(
                    outputOptions->GetTrainDir(),
                    NCatboostOptions::GetExperimentName(featureSetIdx, experimentIdx)
                );
                outputOptions->SetTrainDir(trainDir);
                outputOptions->SetSnapshotFilename(catboostOptions->ModelBasedEvalOptions->BaselineModelSnapshot.Get());
            };
            for (featureSetIdx = 0; featureSetIdx < features.size(); ++featureSetIdx) {
                TSet<ui32> ignoredFeatures = allEvaluatedFeatures;
                for (ui32 feature : features[featureSetIdx]) {
                    if (FeaturesManager.HasBorders(feature)) {
                        ignoredFeatures.erase(feature);
                    } else {
                        CATBOOST_WARNING_LOG << "Ignoring constant feature " << feature  << " in feature set " << featureSetIdx << Endl;
                    }
                }
                TBinarizedFeaturesManager featureManager(FeaturesManager, {ignoredFeatures.begin(), ignoredFeatures.end()});
                if (featureManager.GetDataProviderFeatureIds().empty() || allEvaluatedFeatures == ignoredFeatures) {
                    CATBOOST_WARNING_LOG << "Feature set " << featureSetIdx
                        << " is not evaluated because it consists of ignored or constant features" << Endl;
                    continue;
                }
                auto inputData = CreateInputData(permutationCount, &featureManager);
                auto weak = MakeWeakLearner<TWeakLearner>(featureManager, Config, CatBoostOptions, Random);

                baseCursors->CopyFrom(*startingBaseCursors);
                for (experimentIdx = 0; experimentIdx < ModelBasedEvalConfig.ExperimentCount; ++experimentIdx) {
                    auto metricSaver = ProgressTracker->Clone(forceMetricSaveFunc);
                    TVector<TEnsemble> ignoredModels(permutationCount);
                    auto experimentCursors = CreateCursors(*inputData);
                    experimentCursors->CopyFrom(*baseCursors);
                    BaseIterationSeed = savedBaseSeed + getExperimentStart(experimentIdx);
                    Fit(inputData->DataSets,
                        inputData->GetEstimationPermutation(),
                        GetConstPointers(inputData->Targets),
                        inputData->TestTarget.Get(),
                        &weak,
                        metricSaver.Get(),
                        &(experimentCursors->Cursors),
                        TestDataProvider ? &(experimentCursors->TestCursor) : nullptr,
                        &ignoredModels,
                        experimentCursors->BestTestCursor.Get()
                    );
                    AppendEnsembles(
                        baseInputData->DataSets,
                        baseModels,
                        baseInputData->GetEstimationPermutation(),
                        /*iterStart*/ getExperimentStart(experimentIdx),
                        /*iterEnd*/ getExperimentStart(experimentIdx + 1),
                        &weak,
                        &baseCursors->Cursors,
                        TestDataProvider ? &baseCursors->TestCursor : nullptr
                    );
                }
            } // for indexSet
            BaseIterationSeed = savedBaseSeed;
        }
    };
}
