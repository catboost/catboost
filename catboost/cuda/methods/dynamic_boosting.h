#pragma once

#include "dynamic_boosting_progress.h"
#include "boosting_listeners.h"
#include "learning_rate.h"
#include "random_score_helper.h"

#include <catboost/libs/overfitting_detector/overfitting_detector.h>
#include <catboost/cuda/targets/target_func.h>
#include <catboost/cuda/targets/mse.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset_builder.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <util/stream/format.h>
#include <catboost/libs/options/boosting_options.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/libs/helpers/interrupt.h>

namespace NCatboostCuda {
    template <template <class TMapping, class> class TTargetTemplate,
              class TWeakLearner_,
              NCudaLib::EPtrType CatFeaturesStoragePtrType = NCudaLib::EPtrType::CudaDevice>
    class TDynamicBoosting {
    public:
        using TObjective = TTargetTemplate<NCudaLib::TMirrorMapping, TFeatureParallelDataSet<CatFeaturesStoragePtrType>>;
        using TWeakLearner = TWeakLearner_;
        using TResultModel = TAdditiveModel<typename TWeakLearner::TResultModel>;
        using TWeakModel = typename TWeakLearner::TResultModel;
        using TWeakModelStructure = typename TWeakLearner::TWeakModelStructure;
        using TVec = typename TObjective::TVec;
        using TConstVec = typename TObjective::TConstVec;
        using IListener = IBoostingListener<TObjective, TWeakModel>;

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const TDataProvider* DataProvider;
        const TDataProvider* TestDataProvider;

        TRandom& Random;
        TWeakLearner& Weak;
        const NCatboostOptions::TBoostingOptions& Config;
        const NCatboostOptions::TLossDescription& TargetOptions;

        TVector<IListener*> LearnListeners;
        TVector<IListener*> TestListeners;
        IOverfittingDetector* Detector = nullptr;
        THolder<TSnapshotMeta> SnapshotMeta;

        inline bool Stop(const ui32 iteration) {
            return iteration >= Config.IterationCount || (Detector && Detector->IsNeedStop());
        }

    private:
        struct TFold {
            TSlice EstimateSamples;
            TSlice QualityEvaluateSamples;
        };

        class TPermutationTarget {
        public:
            TPermutationTarget() = default;

            explicit TPermutationTarget(TVector<THolder<TObjective>>&& targets)
                : Targets(std::move(targets))
            {
            }

            const TObjective& GetTarget(ui32 permutationId) const {
                return *Targets[permutationId];
            }

        private:
            TVector<THolder<TObjective>> Targets;
        };

        template <class TData>
        struct TFoldAndPermutationStorage {
            TFoldAndPermutationStorage() = default;

            TFoldAndPermutationStorage(TVector<TVector<TData>>&& foldData,
                                       TData&& estimationData)
                : FoldData(std::move(foldData))
                , Estimation(std::move(estimationData))
            {
            }

            TData& Get(ui32 permutationId, ui32 foldId) {
                return FoldData.at(permutationId).at(foldId);
            }

            TVector<TVector<TData>> FoldData;
            TData Estimation;

            template <class TFunc>
            inline void Foreach(TFunc&& func) {
                for (auto& foldEntries : FoldData) {
                    for (auto& foldEntry : foldEntries) {
                        func(foldEntry);
                    }
                }
                func(Estimation);
            }
        };

    private:
        ui32 GetPermutationBlockSize(ui32 sampleCount) const {
            ui32 suggestedBlockSize = Config.PermutationBlockSize;
            if (sampleCount < 50000) {
                return 1;
            }
            if (suggestedBlockSize > 1) {
                suggestedBlockSize = 1 << IntLog2(suggestedBlockSize);
                while (suggestedBlockSize * 128 > sampleCount) {
                    suggestedBlockSize >>= 1;
                }
            }
            return suggestedBlockSize;
        }

        TFeatureParallelDataSetsHolder<CatFeaturesStoragePtrType> CreateDataSet() const {
            CB_ENSURE(DataProvider);
            ui32 permutationBlockSize = GetPermutationBlockSize(DataProvider->GetSampleCount());

            TFeatureParallelDataSetHoldersBuilder<CatFeaturesStoragePtrType> dataSetsHolderBuilder(FeaturesManager,
                                                                                                   *DataProvider,
                                                                                                   TestDataProvider,
                                                                                                   permutationBlockSize);

            const auto permutationCount = DataProvider->HasTime() ? 1 : Config.PermutationCount;
            return dataSetsHolderBuilder.BuildDataSet(permutationCount);
        }

        TPermutationTarget CreateTargets(const TFeatureParallelDataSetsHolder<CatFeaturesStoragePtrType>& dataSets) const {
            TVector<THolder<TObjective>> targets;
            for (ui32 i = 0; i < dataSets.PermutationsCount(); ++i) {
                targets.push_back(CreateTarget(dataSets.GetDataSetForPermutation(i)));
            }
            return TPermutationTarget(std::move(targets));
        }

        THolder<TObjective> CreateTarget(const TFeatureParallelDataSet<CatFeaturesStoragePtrType>& dataSet) const {
            auto slice = dataSet.GetSamplesMapping().GetObjectsSlice();
            CB_ENSURE(slice.Size());
            return new TObjective(dataSet,
                                  Random,
                                  slice,
                                  TargetOptions);
        }

        inline ui32 MinEstimationSize(ui32 docCount) const {
            if (docCount < 500) {
                return 1;
            }
            const ui32 maxFolds = 18;
            const ui32 folds = IntLog2(NHelpers::CeilDivide(docCount, Config.MinFoldSize));
            if (folds >= maxFolds) {
                return NHelpers::CeilDivide(docCount, 1 << maxFolds);
            }
            return Min<ui32>(Config.MinFoldSize, docCount / 50);
        }

        TVector<TFold> CreateFolds(ui32 sampleCount,
                                   double growthRate,
                                   const IQueriesGrouping& samplesGrouping) const {
            const ui32 minEstimationSize = samplesGrouping.NextQueryOffsetForLine(MinEstimationSize(sampleCount));
            CB_ENSURE(minEstimationSize, "Error: min learn size should be positive");
            CB_ENSURE(growthRate > 1.0, "Error: grow rate should be > 1.0");

            TVector<TFold> folds;
            if (Config.BoostingType == EBoostingType::Plain) {
                folds.push_back({TSlice(0, sampleCount),
                                 TSlice(0, sampleCount)});
                return folds;
            }

            {
                const ui32 testEnd = samplesGrouping.NextQueryOffsetForLine(
                    Min(static_cast<ui32>(minEstimationSize * growthRate), sampleCount));
                folds.push_back({TSlice(0, minEstimationSize), TSlice(minEstimationSize, testEnd)});
            }

            while (folds.back().QualityEvaluateSamples.Right < sampleCount) {
                TSlice learnSlice = TSlice(0, folds.back().QualityEvaluateSamples.Right);
                const ui32 end = samplesGrouping.NextQueryOffsetForLine(
                    Min(static_cast<ui32>(folds.back().QualityEvaluateSamples.Right * growthRate), sampleCount));
                TSlice testSlice = TSlice(folds.back().QualityEvaluateSamples.Right,
                                          end);
                folds.push_back({learnSlice, testSlice});
            }
            return folds;
        }

        inline TVector<TFold> CreateFolds(const TObjective& target,
                                          const TFeatureParallelDataSet<CatFeaturesStoragePtrType>& dataSet,
                                          double growthRate) const {
            Y_UNUSED(target);
            return CreateFolds(static_cast<ui32>(dataSet.GetDataProvider().GetSampleCount()), growthRate,
                               dataSet.GetSamplesGrouping());
        }

        using TCursor = TFoldAndPermutationStorage<TVec>;

        //don't look ahead boosting
        void Fit(const TFeatureParallelDataSetsHolder<CatFeaturesStoragePtrType>& dataSet,
                 const TPermutationTarget& target,
                 const TVector<TVector<TFold>>& permutationFolds,
                 TCursor& cursor,
                 const TObjective* testTarget,
                 TVec* testCursor,
                 TResultModel* result) {
            ui32 iteration = result->Size();
            auto& profiler = NCudaLib::GetProfiler();

            const ui32 permutationCount = dataSet.PermutationsCount();
            CB_ENSURE(permutationCount >= 1);
            const ui32 estimationPermutation = permutationCount - 1;
            const ui32 learnPermutationCount = estimationPermutation ? permutationCount - 1 : 1; //fallback

            const double step = Config.LearningRate;

            auto startTimeBoosting = Now();
            auto lastSnapshotTime = Now();

            {
                for (auto& listener : LearnListeners) {
                    listener->Init(*result,
                                   target.GetTarget(estimationPermutation),
                                   cursor.Estimation);
                }
            }

            if (dataSet.HasTestDataSet()) {
                for (auto& listener : TestListeners) {
                    listener->Init(*result,
                                   *testTarget,
                                   *testCursor);
                }
            }

            while (!Stop(iteration)) {
                CheckInterrupted(); // check after long-lasting operation
                auto iterationTimeGuard = profiler.Profile("Boosting iteration");
                {
                    {
                        //cache
                        THolder<TScopedCacheHolder> iterationCacheHolderPtr;
                        iterationCacheHolderPtr.Reset(new TScopedCacheHolder);

                        auto weakModelStructure = [&]() -> TWeakModelStructure {
                            auto guard = profiler.Profile("Search for weak model structure");
                            const ui32 learnPermutationId = learnPermutationCount > 1
                                                                ? static_cast<const ui32>(Random.NextUniformL() % (learnPermutationCount - 1))
                                                                : 0;

                            const auto& taskTarget = target.GetTarget(learnPermutationId);
                            const auto& taskDataSet = dataSet.GetDataSetForPermutation(learnPermutationId);
                            const auto& taskFolds = permutationFolds[learnPermutationId];

                            using TWeakTarget = TShiftedTargetSlice<TObjective>;

                            auto optimizer = Weak.template CreateStructureSearcher<TWeakTarget, TFeatureParallelDataSet<CatFeaturesStoragePtrType>>(
                                *iterationCacheHolderPtr,
                                taskDataSet);

                            optimizer.SetRandomStrength(CalcScoreModelLengthMult(dataSet.GetDataProvider().GetSampleCount(), iteration * step));

                            if ((Config.BoostingType == EBoostingType::Plain)) {
                                CB_ENSURE(taskFolds.size() == 1);
                                auto allSlice = taskTarget.GetTarget().GetIndices().GetObjectsSlice();
                                TShiftedTargetSlice<TObjective> shiftedTarget(taskTarget,
                                                                              allSlice,
                                                                              cursor.Get(learnPermutationId, 0).ConstCopyView());
                                optimizer.SetTarget(std::move(shiftedTarget));
                            } else {
                                for (ui32 foldId = 0; foldId < taskFolds.size(); ++foldId) {
                                    const auto& fold = taskFolds[foldId];

                                    TShiftedTargetSlice<TObjective> learnTarget(taskTarget,
                                                                                fold.EstimateSamples,
                                                                                cursor.Get(learnPermutationId,
                                                                                           foldId)
                                                                                    .SliceView(fold.EstimateSamples));

                                    TShiftedTargetSlice<TObjective> validateTarget(taskTarget,
                                                                                   fold.QualityEvaluateSamples,
                                                                                   cursor.Get(learnPermutationId, foldId)
                                                                                       .SliceView(fold.QualityEvaluateSamples));

                                    optimizer.AddTask(std::move(learnTarget),
                                                      std::move(validateTarget));
                                }
                            }
                            //search for best model and values of shifted target
                            return optimizer.Fit();
                        }();

                        {
                            auto cacheProfileGuard = profiler.Profile("CacheModelStructure");

                            //should be first for learn-estimation-permutation cache-hit
                            if (dataSet.HasTestDataSet()) {
                                Weak.CacheStructure(*iterationCacheHolderPtr,
                                                    weakModelStructure,
                                                    dataSet.GetTestDataSet());
                            }

                            {
                                const auto& estimationDataSet = dataSet.GetDataSetForPermutation(estimationPermutation);
                                Weak.CacheStructure(*iterationCacheHolderPtr,
                                                    weakModelStructure,
                                                    estimationDataSet);
                            }

                            for (ui32 i = 0; i < learnPermutationCount; ++i) {
                                auto& ds = dataSet.GetDataSetForPermutation(i);
                                Weak.CacheStructure(*iterationCacheHolderPtr,
                                                    weakModelStructure,
                                                    ds);
                            }
                        }

                        TFoldAndPermutationStorage<TWeakModel> models;
                        models.FoldData.resize(learnPermutationCount);

                        {
                            TWeakModel defaultModel(weakModelStructure);
                            for (ui32 permutation = 0; permutation < learnPermutationCount; ++permutation) {
                                models.FoldData[permutation].resize(permutationFolds[permutation].size(), defaultModel);
                            }
                            models.Estimation = defaultModel;
                        }

                        {
                            auto estimateModelsGuard = profiler.Profile("Estimate models");

                            auto estimator = Weak.CreateEstimator();

                            for (ui32 permutation = 0; permutation < learnPermutationCount; ++permutation) {
                                auto& folds = permutationFolds[permutation];

                                for (ui32 foldId = 0; foldId < folds.size(); ++foldId) {
                                    const auto& estimationSlice = folds[foldId].EstimateSamples;

                                    estimator.AddEstimationTask(*iterationCacheHolderPtr,
                                                                TargetSlice(target.GetTarget(permutation), estimationSlice),
                                                                cursor.Get(permutation, foldId).SliceView(estimationSlice),
                                                                &models.FoldData[permutation][foldId]);
                                }
                            }

                            if (!((Config.BoostingType == EBoostingType::Plain) &&
                                  estimationPermutation == 0 /*no avereging permutation case*/))
                            {
                                auto allSlice = dataSet.GetDataSetForPermutation(estimationPermutation).GetIndices().GetObjectsSlice();

                                estimator.AddEstimationTask(*iterationCacheHolderPtr,
                                                            TargetSlice(target.GetTarget(estimationPermutation), allSlice),
                                                            cursor.Estimation.ConstCopyView(),
                                                            &models.Estimation);
                            }
                            estimator.Estimate();
                        }
                        //
                        models.Foreach([&](TWeakModel& model) {
                            model.Rescale(step);
                        });

                        //TODO: make more robust fallback if we disable dontLookAhead
                        if (((Config.BoostingType == EBoostingType::Plain)) && estimationPermutation == 0) {
                            models.Estimation = models.FoldData[0][0];
                        }
                        //
                        {
                            auto appendModelTime = profiler.Profile("Append models time");

                            auto addModelValue = Weak.template CreateAddModelValue<TFeatureParallelDataSet<CatFeaturesStoragePtrType>>(weakModelStructure,
                                                                                                                                       *iterationCacheHolderPtr);

                            if (dataSet.HasTestDataSet()) {
                                addModelValue.AddTask(models.Estimation,
                                                      dataSet.GetTestDataSet(),
                                                      dataSet.GetTestDataSet()
                                                          .GetIndices()
                                                          .ConstCopyView(),
                                                      *testCursor);
                            }

                            addModelValue.AddTask(models.Estimation,
                                                  dataSet.GetDataSetForPermutation(estimationPermutation),
                                                  dataSet.GetDataSetForPermutation(estimationPermutation)
                                                      .GetIndices()
                                                      .ConstCopyView(),
                                                  cursor.Estimation);

                            for (ui32 permutation = 0; permutation < learnPermutationCount; ++permutation) {
                                auto& permutationModels = models.FoldData[permutation];
                                auto& folds = permutationFolds[permutation];

                                const auto& ds = dataSet.GetDataSetForPermutation(permutation);

                                for (ui32 foldId = 0; foldId < folds.size(); ++foldId) {
                                    TFold fold = folds[foldId];
                                    TSlice allSlice = TSlice(0, fold.QualityEvaluateSamples.Right);
                                    CB_ENSURE(cursor.Get(permutation, foldId).GetObjectsSlice() == allSlice);

                                    addModelValue.AddTask(permutationModels[foldId],
                                                          ds,
                                                          ds.GetIndices().SliceView(allSlice),
                                                          cursor.Get(permutation, foldId));
                                }
                            }

                            addModelValue.Proceed();
                        }

                        result->AddWeakModel(models.Estimation);
                    }

                    {
                        auto learnListenerTimeGuard = profiler.Profile("Boosting listeners time: Learn");

                        for (auto& listener : LearnListeners) {
                            listener->Invoke(*result,
                                             target.GetTarget(estimationPermutation),
                                             cursor.Estimation);
                        }
                    }

                    if (dataSet.HasTestDataSet()) {
                        auto testListenerTimeGuard = profiler.Profile("Boosting listeners time: Test");

                        for (auto& listener : TestListeners) {
                            listener->Invoke(*result,
                                             *testTarget,
                                             *testCursor);
                        }
                    }

                    iteration++;
                    if (SnapshotMeta && ((Now() - lastSnapshotTime).SecondsFloat() > SnapshotMeta->SaveIntervalSeconds)) {
                        auto progress = MakeProgress(FeaturesManager, *result, cursor, testCursor);
                        TProgressHelper(GpuProgressLabel()).Write(SnapshotMeta->Path, [&](IOutputStream* out) {
                            ::Save(out, SnapshotMeta->TaskOptions);
                            ::Save(out, progress);
                        });
                        lastSnapshotTime = Now();
                    }
                }
            }
            MATRIXNET_INFO_LOG << "Total time " << (Now() - startTimeBoosting).SecondsFloat() << Endl;
        }

    public:
        TDynamicBoosting(TBinarizedFeaturesManager& binarizedFeaturesManager,
                         const NCatboostOptions::TBoostingOptions& config,
                         const NCatboostOptions::TLossDescription& targetOptions,
                         TRandom& random,
                         TWeakLearner& weak)
            : FeaturesManager(binarizedFeaturesManager)
            , Random(random)
            , Weak(weak)
            , Config(config)
            , TargetOptions(targetOptions)
        {
        }

        virtual ~TDynamicBoosting() = default;

        TDynamicBoosting& SetDataProvider(const TDataProvider& learnData,
                                          const TDataProvider* testData = nullptr) {
            DataProvider = &learnData;
            TestDataProvider = testData;
            return *this;
        }

        TDynamicBoosting& RegisterLearnListener(IListener& listener) {
            LearnListeners.push_back(&listener);
            return *this;
        }

        TDynamicBoosting& RegisterTestListener(IListener& listener) {
            Y_ENSURE(TestDataProvider, "Error: need test set for test listener");
            TestListeners.push_back(&listener);
            return *this;
        }

        TDynamicBoosting& AddOverfitDetector(IOverfittingDetector& detector) {
            Detector = &detector;
            return *this;
        }

        TDynamicBoosting& SaveSnapshot(const TString& snapshotPath, const TString& taskOptions, ui64 snapshotInterval) {
            SnapshotMeta = MakeHolder<TSnapshotMeta>(snapshotPath, taskOptions, snapshotInterval);
            return *this;
        }

        struct TBoostingState {
            TFeatureParallelDataSetsHolder<CatFeaturesStoragePtrType> DataSets;

            TPermutationTarget Targets;
            TCursor Cursor;

            TVec TestCursor;
            THolder<TObjective> TestTarget;

            TVector<TVector<TFold>> PermutationFolds;

            ui32 GetEstimationPermutation() const {
                return DataSets.PermutationsCount() - 1;
            }
        };

        THolder<TBoostingState> CreateState() const {
            THolder<TBoostingState> state(new TBoostingState);
            state->DataSets = CreateDataSet();
            state->Targets = CreateTargets(state->DataSets);

            if (TestDataProvider) {
                state->TestTarget = CreateTarget(state->DataSets.GetTestDataSet());
                state->TestCursor = TMirrorBuffer<float>::CopyMapping(state->DataSets.GetTestDataSet().GetTarget().GetTargets());
                if (TestDataProvider->HasBaseline()) {
                    state->TestCursor.Write(TestDataProvider->GetBaseline());
                }
                FillBuffer(state->TestCursor, 0.0f);
            }

            const ui32 estimationPermutation = state->DataSets.PermutationsCount() - 1;
            const ui32 learnPermutationCount = estimationPermutation ? estimationPermutation
                                                                     : 1; //fallback to 1 permutation to learn and test
            state->PermutationFolds.resize(learnPermutationCount);

            state->Cursor.FoldData.resize(learnPermutationCount);

            for (ui32 i = 0; i < learnPermutationCount; ++i) {
                auto& folds = state->PermutationFolds[i];
                const auto& permutation = state->DataSets.GetPermutation(i);
                TVector<float> baseline;
                if (DataProvider->HasBaseline()) {
                    baseline = permutation.Gather(DataProvider->GetBaseline());
                } else {
                    baseline.resize(DataProvider->GetSampleCount(), 0.0f);
                }

                folds = CreateFolds(state->Targets.GetTarget(i),
                                    state->DataSets.GetDataSetForPermutation(i),
                                    Config.FoldLenMultiplier);

                auto& foldCursors = state->Cursor.FoldData[i];
                foldCursors.resize(folds.size());

                for (ui32 fold = 0; fold < folds.size(); ++fold) {
                    auto mapping = NCudaLib::TMirrorMapping(folds[fold].QualityEvaluateSamples.Right);
                    foldCursors[fold] = TVec::Create(mapping);
                    foldCursors[fold].Write(baseline);
                }
            }
            {
                auto& permutation = state->DataSets.GetPermutation(estimationPermutation);
                TVector<float> baseline;
                if (DataProvider->HasBaseline()) {
                    baseline = permutation.Gather(DataProvider->GetBaseline());
                } else {
                    baseline.resize(DataProvider->GetSampleCount(), 0.0f);
                }

                state->Cursor.Estimation = TMirrorBuffer<float>::CopyMapping(state->DataSets.GetDataSetForPermutation(estimationPermutation).GetTarget().GetTargets());
                state->Cursor.Estimation.Write(baseline);
            }
            return state;
        }

        THolder<TResultModel> Run() {
            auto state = CreateState();
            THolder<TResultModel> resultModel = MakeHolder<TResultModel>();

            if (SnapshotMeta && NFs::Exists(SnapshotMeta->Path)) {
                if (GetFileLength(SnapshotMeta->Path) == 0) {
                    MATRIXNET_WARNING_LOG << "Empty snapshot file. Something possible wrong" << Endl;
                } else {
                    TDynamicBoostingProgress<TResultModel> progress;
                    TProgressHelper(GpuProgressLabel()).CheckedLoad(SnapshotMeta->Path, [&](TIFStream* in) {
                        TString optionsStr;
                        ::Load(in, optionsStr);
                        ::Load(in, progress);
                    });
                    WriteProgressToGpu(progress,
                                       FeaturesManager,
                                       *resultModel,
                                       state->Cursor,
                                       TestDataProvider ? &state->TestCursor : nullptr);
                }
            }

            Fit(state->DataSets,
                state->Targets,
                state->PermutationFolds,
                state->Cursor,
                state->TestTarget.Get(),
                TestDataProvider ? &state->TestCursor : nullptr,
                resultModel.Get());
            return resultModel;
        }
    };
}
