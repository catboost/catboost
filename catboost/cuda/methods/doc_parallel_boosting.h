#pragma once

#include "learning_rate.h"
#include "random_score_helper.h"
#include "doc_parallel_boosting_progress.h"
#include "boosting_progress_tracker.h"

#include <catboost/libs/overfitting_detector/overfitting_detector.h>
#include <catboost/cuda/targets/target_func.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset_builder.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/options/boosting_options.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
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
        using TWeakModelStructure = typename TWeakLearner::TWeakModelStructure;
        using TVec = typename TObjective::TVec;
        using TConstVec = typename TObjective::TConstVec;
        using TDataSet = TDocParallelDataSet;
        using TCursor = TStripeBuffer<float>;

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const TDataProvider* DataProvider = nullptr;
        const TDataProvider* TestDataProvider = nullptr;
        TBoostingProgressTracker* ProgressTracker = nullptr;

        TGpuAwareRandom& Random;
        ui64 BaseIterationSeed;
        TWeakLearner& Weak;
        const NCatboostOptions::TBoostingOptions& Config;
        const NCatboostOptions::TLossDescription& TargetOptions;

    private:
        inline static TDocParallelDataSetsHolder CreateDocParallelDataSet(TBinarizedFeaturesManager& manager,
                                                                          const TDataProvider& dataProvider,
                                                                          const TDataProvider* test,
                                                                          ui32 permutationCount) {
            TDocParallelDataSetBuilder dataSetsHolderBuilder(manager,
                                                             dataProvider,
                                                             test);
            return dataSetsHolderBuilder.BuildDataSet(permutationCount);
        }

        struct TBoostingState {
            using TEnsemble = TAdditiveModel<TWeakModel>;
            using TCursor = TStripeBuffer<float>;
            TDocParallelDataSetsHolder DataSets;

            TVector<TCursor> Cursors;
            TVector<TEnsemble> Models;

            TVec TestCursor;
            THolder<TVec> BestTestCursor;

            ui32 GetEstimationPermutation() const {
                return DataSets.PermutationsCount() - 1;
            }
        };

        THolder<TBoostingState> CreateState(ui32 permutationCount) {
            CB_ENSURE(DataProvider);
            const auto& dataProvider = *DataProvider;
            THolder<TBoostingState> state(new TBoostingState);
            state->DataSets = CreateDocParallelDataSet(FeaturesManager, dataProvider, TestDataProvider, permutationCount);

            state->Cursors.resize(state->DataSets.PermutationsCount());

            for (ui32 i = 0; i < permutationCount; ++i) {
                const auto& loadBalancingPermutation = state->DataSets.GetLoadBalancingPermutation();
                state->Cursors[i].Reset(state->DataSets.GetDataSetForPermutation(0).GetTarget().GetSamplesMapping());
                CB_ENSURE(state->Cursors[i].GetMapping().GetObjectsSlice().Size());

                if (dataProvider.HasBaseline()) {
                    TVector<float> baseline = loadBalancingPermutation.Gather(dataProvider.GetBaseline());
                    CB_ENSURE(baseline.size() == state->Cursors[i].GetObjectsSlice().Size());
                    state->Cursors[i].Write(baseline);
                } else {
                    FillBuffer(state->Cursors[i], 0.0f);
                }
            }

            if (TestDataProvider) {
                state->TestCursor = TCursor::CopyMapping(state->DataSets.GetTestDataSet().GetTarget().GetTargets());
                if (TestDataProvider->HasBaseline()) {
                    const auto& testPermutation = state->DataSets.GetTestLoadBalancingPermutation();
                    auto baseline = testPermutation.Gather(TestDataProvider->GetBaseline());
                    state->TestCursor.Write(baseline);
                } else {
                    FillBuffer(state->TestCursor, 0.0f);
                }
            }

            state->Models.resize(permutationCount);

            if (ProgressTracker->NeedBestTestCursor()) {
                Y_VERIFY(TestDataProvider);
                state->BestTestCursor = new TStripeBuffer<float>();
                (*state->BestTestCursor).Reset(state->TestCursor.GetMapping());
            }

            ProgressTracker->MaybeRestoreFromSnapshot([&](IInputStream* in) {
                using TProgress = TBoostingProgress<TResultModel>;
                TProgress progress;
                ::Load(in, progress);
                if (state->BestTestCursor) {
                    LoadCudaBuffer(in, state->BestTestCursor.Get());
                }
                state->Models = RestoreFromProgress(FeaturesManager, progress);

                CB_ENSURE(state->Models.size() == permutationCount, "Progress permutation count differs from current learning task: " << state->Models.size() << " / " << permutationCount);
                {
                    auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile("Restore from progress");
                    AppendEnsembles(state->DataSets,
                                    state->Models,
                                    state->GetEstimationPermutation(),
                                    &state->Cursors,
                                    TestDataProvider ? &state->TestCursor : nullptr);
                }
                MATRIXNET_DEBUG_LOG << "Restore #" << state->Models[0].Size() << " trees from progress" << Endl;
            });

            return state;
        }

        void AppendEnsembles(const TDocParallelDataSetsHolder& dataSets,
                             const TVector<TResultModel>& ensembles,
                             ui32 estimationPermutation,
                             TVector<TVec>* cursors,
                             TVec* testCursor) {
            TVector<TWeakModel> iterationWeakModels;
            iterationWeakModels.resize(ensembles.size());
            for (ui32 iter = 0; iter < ensembles[0].Size(); ++iter) {
                for (ui32 permutation = 0; permutation < ensembles.size(); ++permutation) {
                    iterationWeakModels[permutation] = ensembles[permutation][iter];
                }
                AppendModels(dataSets, iterationWeakModels, estimationPermutation, cursors, testCursor);
            }
        }

        void AppendModels(const TDocParallelDataSetsHolder& dataSets,
                          const TVector<TWeakModel>& iterationsModels,
                          ui32 estimationPermutation,
                          TVector<TCursor>* cursors,
                          TCursor* testCursor) {
            const bool streamParallelAppend = false;
            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();
            auto appendModelTime = profiler.Profile("Append models time");
            auto addModelValue = Weak.template CreateAddModelValue<TDocParallelDataSet>(streamParallelAppend);

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
            return new TObjective(dataSet,
                                  Random,
                                  TargetOptions);
        }

        //TODO(noxoomo): remove overhead of multiple target for permutation datasets
        //don't look ahead boosting
        void Fit(const TDocParallelDataSetsHolder& dataSet,
                 const ui32 estimationPermutation,
                 const TVector<const TObjective*>& learnTarget,
                 const TObjective* testTarget,
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

            TMetricCalcer<TObjective> learnMetricCalcer(*learnTarget[estimationPermutation]);
            THolder<TMetricCalcer<TObjective>> testMetricCalcer;
            if (testTarget) {
                testMetricCalcer = new TMetricCalcer<TObjective>(*testTarget);
            }

            while (!(progressTracker->ShouldStop())) {
                CheckInterrupted(); // check after long-lasting operation
                auto iterationTimeGuard = profiler.Profile("Boosting iteration");
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
                    auto target = TTargetAtPointTrait<TObjective>::Create(*(learnTarget[learnPermutationId]),
                                                                          (*learnCursors)[learnPermutationId]);
                    auto mult = CalcScoreModelLengthMult(dataSet.GetDataProvider().GetSampleCount(),
                                                         iteration * step);
                    auto optimizer = Weak.template CreateStructureSearcher<TWeakTarget, TDocParallelDataSet>(mult);
                    //search for best model and values of shifted target
                    auto model = optimizer.Fit(taskDataSet,
                                               target);
                    TVector<TWeakModel> models;
                    models.resize(result->size(), model);
                    return models;
                }();

                if (Weak.NeedEstimation()) {
                    auto estimateModelsGuard = profiler.Profile("Estimate models");
                    auto estimator = Weak.CreateEstimator();

                    for (ui32 permutation = 0; permutation < permutationCount; ++permutation) {
                        const auto& taskDataSet = dataSet.GetDataSetForPermutation(permutation);
                        estimator.AddEstimationTask(*(learnTarget[permutation]),
                                                    taskDataSet,
                                                    (*learnCursors)[permutation],
                                                    &iterationModels[permutation]);
                    }
                    estimator.Estimate();
                }
                //
                for (auto& iterationModel : iterationModels) {
                    iterationModel.Rescale(step);
                };

                AppendModels(dataSet, iterationModels,
                             estimationPermutation,
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
                    Y_VERIFY(testCursor);
                    bestTestCursor->Copy(*testCursor);
                }

                progressTracker->MaybeSaveSnapshot([&](IOutputStream* out) {
                    auto progress = MakeProgress(FeaturesManager, *result);
                    ::Save(out, progress);
                    if (bestTestCursor) {
                        SaveCudaBuffer(*bestTestCursor, out);
                    }
                });
            }

            if (bestTestCursor) {
                TVector<float> bestTestCpu;
                bestTestCursor->Read(bestTestCpu);
                progressTracker->SetBestTestCursor(bestTestCpu);
            }
            MATRIXNET_INFO_LOG << "Total time " << (Now() - startTimeBoosting).SecondsFloat() << Endl;
        }

    public:
        TBoosting(TBinarizedFeaturesManager& binarizedFeaturesManager,
                  const NCatboostOptions::TBoostingOptions& config,
                  const NCatboostOptions::TLossDescription& targetOptions,
                  EGpuCatFeaturesStorage,
                  TGpuAwareRandom& random,
                  TWeakLearner& weak)
            : FeaturesManager(binarizedFeaturesManager)
            , Random(random)
            , BaseIterationSeed(random.NextUniformL())
            , Weak(weak)
            , Config(config)
            , TargetOptions(targetOptions)
        {
        }

        virtual ~TBoosting() = default;

        //TODO(noxoomo): to common with dynamic boosting superclass
        TBoosting& SetDataProvider(const TDataProvider& learnData,
                                   const TDataProvider* testData = nullptr) {
            DataProvider = &learnData;
            TestDataProvider = testData;
            return *this;
        }

        void SetBoostingProgressTracker(TBoostingProgressTracker* progressTracker) {
            ProgressTracker = progressTracker;
        }

        THolder<TResultModel> Run() {
            CB_ENSURE(DataProvider, "Error: set dataProvider first");
            CB_ENSURE(ProgressTracker, "Error: set boosting tracker first");
            auto state = CreateState(Config.PermutationCount);

            TVector<THolder<TObjective>> targets;
            TVector<const TObjective*> constTargets;
            for (ui32 i = 0; i < state->Cursors.size(); ++i) {
                targets.push_back(CreateTarget(state->DataSets.GetDataSetForPermutation(i)));
                constTargets.push_back(targets.back().Get());
            }

            THolder<TObjective> testTarget;
            if (TestDataProvider) {
                testTarget = CreateTarget(state->DataSets.GetTestDataSet());
            }

            Fit(state->DataSets,
                state->GetEstimationPermutation(),
                constTargets,
                testTarget.Get(),
                ProgressTracker,
                &(state->Cursors),
                TestDataProvider ? &(state->TestCursor) : nullptr,
                &state->Models,
                state->BestTestCursor.Get());

            return new TResultModel(state->Models[state->GetEstimationPermutation()]);
        }
    };
}
