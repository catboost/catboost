#pragma once

#include "dynamic_boosting_progress.h"
#include "learning_rate.h"
#include "random_score_helper.h"
#include "boosting_progress_tracker.h"

#include <catboost/cuda/targets/target_func.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset_builder.h>

#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/helpers/math_utils.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/metrics/optimal_const_for_loss.h>
#include <catboost/private/libs/options/boosting_options.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/libs/overfitting_detector/overfitting_detector.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/stream/format.h>

namespace NCatboostCuda {
    namespace {
        template <typename TTargetSlice>
        inline bool IsMetricDefined(ELossFunction lossFunction, const TTargetSlice& targetSlice) {
            if (IsPairwiseMetric(lossFunction) && !targetSlice.GetSamplesGrouping().HasPairs()) {
                CATBOOST_DEBUG_LOG << "Dynamic boosting skipped some fold for some permutation because it did not have pairs; "
                    "this may happen if train or test datasets are very small" << Endl;
                return false;
            }
            if (IsGroupwiseMetric(lossFunction)) {
                const auto objectCount = targetSlice.GetTarget().GetSamplesMapping().GetObjectsSlice().Size();
                if (targetSlice.GetSamplesGrouping().GetQueryCount() >= objectCount) {
                    CATBOOST_DEBUG_LOG << "Dynamic boosting skipped some fold for some permutation because all groups were trivial; "
                        "this may happen if train or test datasets are very small" << Endl;
                    return false;
                }
            }
            return true;
        }
    }
    template <template <class TMapping> class TTargetTemplate,
              class TWeakLearner_>
    class TDynamicBoosting {
    public:
        using TObjective = TTargetTemplate<NCudaLib::TMirrorMapping>;
        using TWeakLearner = TWeakLearner_;
        using TResultModel = TAdditiveModel<typename TWeakLearner::TResultModel>;
        using TWeakModel = typename TWeakLearner::TResultModel;
        using TWeakModelStructure = typename TWeakLearner::TWeakModelStructure;
        using TVec = typename TObjective::TVec;
        using TConstVec = typename TObjective::TConstVec;

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        const NCB::TTrainingDataProvider* DataProvider = nullptr;
        const NCB::TFeatureEstimators* Estimators;
        const NCB::TTrainingDataProvider* TestDataProvider = nullptr;
        TBoostingProgressTracker* ProgressTracker = nullptr;

        EGpuCatFeaturesStorage CatFeaturesStorage;
        TGpuAwareRandom& Random;
        const NCatboostOptions::TCatBoostOptions& CatBoostOptions;
        const NCatboostOptions::TBoostingOptions& Config;
        const NCatboostOptions::TLossDescription& TargetOptions;
        const TMaybe<TCustomObjectiveDescriptor>& ObjectiveDescriptor;

        NPar::ILocalExecutor* LocalExecutor;

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
                suggestedBlockSize = 1 << NCB::IntLog2(suggestedBlockSize);
                while (suggestedBlockSize * 128 > sampleCount) {
                    suggestedBlockSize >>= 1;
                }
            }
            return suggestedBlockSize;
        }

        TFeatureParallelDataSetsHolder CreateDataSet() const {
            CB_ENSURE(DataProvider);
            ui32 permutationBlockSize = GetPermutationBlockSize(DataProvider->GetObjectCount());

            TFeatureParallelDataSetHoldersBuilder dataSetsHolderBuilder(FeaturesManager,
                                                                        *DataProvider,
                                                                        *Estimators,
                                                                        TestDataProvider,
                                                                        permutationBlockSize,
                                                                        CatFeaturesStorage

            );

            const auto permutationCount = DataProvider->ObjectsData->GetOrder() == NCB::EObjectsOrder::Ordered ? 1
                                                                                                               : Config.PermutationCount;
            return dataSetsHolderBuilder.BuildDataSet(permutationCount, LocalExecutor);
        }

        TPermutationTarget CreateTargets(const TFeatureParallelDataSetsHolder& dataSets) const {
            TVector<THolder<TObjective>> targets;
            for (ui32 i = 0; i < dataSets.PermutationsCount(); ++i) {
                targets.push_back(CreateTarget(dataSets.GetDataSetForPermutation(i)));
            }
            return TPermutationTarget(std::move(targets));
        }

        THolder<TObjective> CreateTarget(const TFeatureParallelDataSet& dataSet) const {
            auto slice = dataSet.GetSamplesMapping().GetObjectsSlice();
            CB_ENSURE(slice.Size());
            return MakeHolder<TObjective>(dataSet,
                                  Random,
                                  slice,
                                  TargetOptions,
                                  ObjectiveDescriptor);
        }

        inline ui32 MinEstimationSize(ui32 docCount) const {
            if (docCount < 500) {
                return 1;
            }
            const ui32 maxFolds = 18;
            const ui32 folds = NCB::IntLog2(NHelpers::CeilDivide(docCount, Config.MinFoldSize));
            if (folds >= maxFolds) {
                return NHelpers::CeilDivide(docCount, 1 << maxFolds);
            }
            return Min<ui32>(Config.MinFoldSize, docCount / 50);
        }

        TVector<TFold> CreateFolds(ui32 sampleCount,
                                   double growthRate,
                                   const IQueriesGrouping& samplesGrouping) const {
            ui32 minEstimationSize = samplesGrouping.NextQueryOffsetForLine(MinEstimationSize(sampleCount));
            const ui32 devCount = NCudaLib::GetCudaManager().GetDeviceCount();
            //we should have at least several queries per devices
            if (devCount > 1) {
                minEstimationSize = Max(minEstimationSize,
                                        samplesGrouping.GetQueryOffset(Min<ui32>(16 * devCount, samplesGrouping.GetQueryCount() / 2)));
            }

            CB_ENSURE(samplesGrouping.GetQueryCount() >= 4 * devCount, "Error: pool has just " << samplesGrouping.GetQueryCount() << " groups or docs, can't use #" << devCount << " GPUs to learn on such small pool");
            CB_ENSURE(minEstimationSize, "Error: min learn size should be positive");
            CB_ENSURE(growthRate > 1.0, "Error: grow rate should be > 1.0");

            TVector<TFold> folds;
            if (Config.BoostingType == EBoostingType::Plain) {
                folds.push_back({TSlice(0, sampleCount),
                                 TSlice(0, sampleCount)});
                return folds;
            }

            {
                const ui32 testEnd = samplesGrouping.NextQueryOffsetForLine(Min(static_cast<ui32>(minEstimationSize * growthRate), sampleCount));
                folds.push_back({TSlice(0, minEstimationSize), TSlice(minEstimationSize, testEnd)});
            }

            while (folds.back().QualityEvaluateSamples.Right < sampleCount) {
                TSlice learnSlice = TSlice(0, folds.back().QualityEvaluateSamples.Right);
                const ui32 end = samplesGrouping.NextQueryOffsetForLine(Min(static_cast<ui32>(folds.back().QualityEvaluateSamples.Right * growthRate), sampleCount));
                TSlice testSlice = TSlice(folds.back().QualityEvaluateSamples.Right,
                                          end);
                folds.push_back({learnSlice, testSlice});
            }
            return folds;
        }

        inline TVector<TFold> CreateFolds(const TObjective& target,
                                          const TFeatureParallelDataSet& dataSet,
                                          double growthRate) const {
            Y_UNUSED(target);
            return CreateFolds(static_cast<ui32>(dataSet.GetDataProvider().GetObjectCount()), growthRate,
                               dataSet.GetSamplesGrouping());
        }

        using TCursor = TFoldAndPermutationStorage<TVec>;

        //don't look ahead boosting
        void Fit(const TFeatureParallelDataSetsHolder& dataSet,
                 const TPermutationTarget& target,
                 const TVector<TVector<TFold>>& permutationFolds,
                 const TObjective* testTarget,
                 TBoostingProgressTracker* progressTracker,
                 TCursor* cursorPtr,
                 TVec* testCursor,
                 TVec* bestTestCursor,
                 TResultModel* result) {
            auto& cursor = *cursorPtr;
            auto& profiler = NCudaLib::GetProfiler();

            const ui32 permutationCount = dataSet.PermutationsCount();
            CB_ENSURE(permutationCount >= 1);
            const ui32 estimationPermutation = permutationCount - 1;
            const ui32 learnPermutationCount = estimationPermutation ? permutationCount - 1 : 1; //fallback

            const double step = Config.LearningRate;
            auto startTimeBoosting = Now();

            TMetricCalcer<TObjective> metricCalcer(target.GetTarget(estimationPermutation), LocalExecutor);
            THolder<TMetricCalcer<TObjective>> testMetricCalcer;
            if (testTarget) {
                testMetricCalcer = MakeHolder<TMetricCalcer<TObjective>>(*testTarget, LocalExecutor);
            }

            auto snapshotSaver = [&](IOutputStream* out) {
                auto progress = MakeProgress(FeaturesManager, *result, cursor, testCursor);
                ::Save(out, progress);
                if (bestTestCursor) {
                    SaveCudaBuffer(*bestTestCursor, out);
                }
            };

            auto weak = MakeWeakLearner<TWeakLearner>(FeaturesManager, Config, CatBoostOptions, Random);
            while (!progressTracker->ShouldStop()) {
                CheckInterrupted(); // check after long-lasting operation
                auto iterationTimeGuard = profiler.Profile("Boosting iteration");
                progressTracker->MaybeSaveSnapshot(snapshotSaver);
                TOneIterationProgressTracker iterationProgressTracker(*progressTracker);
                const ui32 iteration = iterationProgressTracker.GetCurrentIteration();
                {
                    //cache
                    THolder<TScopedCacheHolder> iterationCacheHolderPtr;
                    iterationCacheHolderPtr.Reset(new TScopedCacheHolder);

                    auto weakModelStructure = [&]() -> TWeakModelStructure {
                        auto guard = profiler.Profile("Search for weak model structure");
                        const ui32 learnPermutationId = learnPermutationCount > 1
                                                            ? static_cast<const ui32>(Random.NextUniformL() %
                                                                                      (learnPermutationCount - 1))
                                                            : 0;

                        const auto& taskTarget = target.GetTarget(learnPermutationId);
                        const auto& taskDataSet = dataSet.GetDataSetForPermutation(learnPermutationId);
                        const auto& taskFolds = permutationFolds[learnPermutationId];

                        using TWeakTarget = typename TTargetAtPointTrait<TObjective>::Type;

                        auto optimizer = weak.template CreateStructureSearcher<TWeakTarget, TFeatureParallelDataSet>(
                            *iterationCacheHolderPtr,
                            taskDataSet,
                            *result);

                        optimizer.SetRandomStrength(
                            CalcScoreModelLengthMult(dataSet.GetDataProvider().GetObjectCount(),
                                                     iteration * step));

                        if ((Config.BoostingType == EBoostingType::Plain)) {
                            CB_ENSURE(taskFolds.size() == 1);
                            auto allSlice = taskTarget.GetTarget().GetIndices().GetObjectsSlice();
                            auto shiftedTarget = TTargetAtPointTrait<TObjective>::Create(taskTarget, allSlice,
                                                                                         cursor.Get(
                                                                                                   learnPermutationId,
                                                                                                   0)
                                                                                             .ConstCopyView());
                            optimizer.SetTarget(std::move(shiftedTarget));
                        } else {
                            for (ui32 foldId = 0; foldId < taskFolds.size(); ++foldId) {
                                const auto& fold = taskFolds[foldId];
                                auto learnTarget = TTargetAtPointTrait<TObjective>::Create(
                                    taskTarget,
                                    fold.EstimateSamples,
                                    cursor.Get(learnPermutationId, foldId).SliceView(fold.EstimateSamples).AsConstBuf()
                                );
                                auto validateTarget = TTargetAtPointTrait<TObjective>::Create(
                                    taskTarget,
                                    fold.QualityEvaluateSamples,
                                    cursor.Get(learnPermutationId, foldId).SliceView(fold.QualityEvaluateSamples).AsConstBuf()
                                );

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
                            weak.CacheStructure(*iterationCacheHolderPtr,
                                                weakModelStructure,
                                                dataSet.GetTestDataSet());
                        }

                        {
                            const auto& estimationDataSet = dataSet.GetDataSetForPermutation(estimationPermutation);
                            weak.CacheStructure(*iterationCacheHolderPtr,
                                                weakModelStructure,
                                                estimationDataSet);
                        }

                        for (ui32 i = 0; i < learnPermutationCount; ++i) {
                            auto& ds = dataSet.GetDataSetForPermutation(i);
                            weak.CacheStructure(*iterationCacheHolderPtr,
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

                        auto estimator = weak.CreateEstimator();

                        for (ui32 permutation = 0; permutation < learnPermutationCount; ++permutation) {
                            auto& folds = permutationFolds[permutation];
                            const auto& permutationDataSet = dataSet.GetDataSetForPermutation(permutation);

                            for (ui32 foldId = 0; foldId < folds.size(); ++foldId) {
                                const auto& estimationSlice = folds[foldId].EstimateSamples;

                                const auto& targetPermutation = target.GetTarget(permutation);
                                const auto& targetSlice = TargetSlice(targetPermutation, estimationSlice);
                                if (!IsMetricDefined(targetPermutation.GetType(), targetSlice)) {
                                    continue;
                                }
                                const auto& cursorSlice = cursor.Get(permutation, foldId).SliceView(estimationSlice);
                                estimator.AddEstimationTask(*iterationCacheHolderPtr,
                                                            targetSlice,
                                                            permutationDataSet,
                                                            cursorSlice.AsConstBuf(),
                                                            &models.FoldData[permutation][foldId]);
                            }
                        }

                        if (!((Config.BoostingType == EBoostingType::Plain) && estimationPermutation == 0 /*no avereging permutation case*/)) {
                            auto allSlice = dataSet.GetDataSetForPermutation(
                                                       estimationPermutation)
                                                .GetIndices()
                                                .GetObjectsSlice();

                            estimator.AddEstimationTask(*iterationCacheHolderPtr,
                                                        TargetSlice(target.GetTarget(estimationPermutation),
                                                                    allSlice),
                                                        dataSet.GetDataSetForPermutation(estimationPermutation),
                                                        cursor.Estimation.ConstCopyView(),
                                                        &models.Estimation);
                        }
                        estimator.Estimate(LocalExecutor);
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

                        auto addModelValue = weak.template CreateAddModelValue<TFeatureParallelDataSet>(
                            weakModelStructure,
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
                    metricCalcer.SetPoint(cursor.Estimation.ConstCopyView());
                    iterationProgressTracker.TrackLearnErrors(metricCalcer);
                }

                if (dataSet.HasTestDataSet()) {
                    auto testListenerTimeGuard = profiler.Profile("Boosting listeners time: Test");

                    testMetricCalcer->SetPoint(testCursor->ConstCopyView());
                    iterationProgressTracker.TrackTestErrors(*testMetricCalcer);
                }

                if (iterationProgressTracker.IsBestTestIteration() && bestTestCursor) {
                    CB_ENSURE(testCursor, "Need cursor for test data");
                    bestTestCursor->Copy(*testCursor);
                }
            }

            progressTracker->MaybeSaveSnapshot(snapshotSaver);

            if (bestTestCursor) {
                TVector<TVector<double>> cpuApprox;
                ReadApproxInCpuFormat(*bestTestCursor, TargetOptions.GetLossFunction() == ELossFunction::MultiClass, &cpuApprox);
                progressTracker->SetBestTestCursor(std::move(cpuApprox));
            }
            if (dataSet.HasTestDataSet()) {
                TVector<TVector<double>> cpuApprox;
                ReadApproxInCpuFormat(*testCursor, TargetOptions.GetLossFunction() == ELossFunction::MultiClass, &cpuApprox);
                progressTracker->SetFinalTestCursor(std::move(cpuApprox));
            }
            CATBOOST_INFO_LOG << "Total time " << (Now() - startTimeBoosting).SecondsFloat() << Endl;
        }

    public:
        TDynamicBoosting(TBinarizedFeaturesManager& binarizedFeaturesManager,
                         const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                         const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                         EGpuCatFeaturesStorage catFeaturesStorage,
                         TGpuAwareRandom& random,
                         NPar::ILocalExecutor* localExecutor)
            : FeaturesManager(binarizedFeaturesManager)
            , CatFeaturesStorage(catFeaturesStorage)
            , Random(random)
            , CatBoostOptions(catBoostOptions)
            , Config(catBoostOptions.BoostingOptions)
            , TargetOptions(catBoostOptions.LossFunctionDescription)
            , ObjectiveDescriptor(objectiveDescriptor)
            , LocalExecutor(localExecutor)
        {
        }

        virtual ~TDynamicBoosting() = default;

        TDynamicBoosting& SetDataProvider(const NCB::TTrainingDataProvider& learnData,
                                          const NCB::TFeatureEstimators& estimators,
                                          const NCB::TTrainingDataProvider* testData = nullptr) {
            DataProvider = &learnData;
            Estimators = &estimators;
            TestDataProvider = testData;
            return *this;
        }

        void SetBoostingProgressTracker(TBoostingProgressTracker* progressTracker) {
            ProgressTracker = progressTracker;
        }

        struct TBoostingState {
            TFeatureParallelDataSetsHolder DataSets;

            TPermutationTarget Targets;
            TCursor Cursor;

            TVec TestCursor;
            THolder<TObjective> TestTarget;

            TVector<TVector<TFold>> PermutationFolds;

            THolder<TVec> BestTestCursor;
            TMaybe<TVector<double>> StartingPoint;

            ui32 GetEstimationPermutation() const {
                return DataSets.PermutationsCount() - 1;
            }
        };

        THolder<TBoostingState> CreateState() const {
            THolder<TBoostingState> state(new TBoostingState);
            state->DataSets = CreateDataSet();
            state->Targets = CreateTargets(state->DataSets);

            const bool isBoostFromAverage = CatBoostOptions.BoostingOptions->BoostFromAverage.Get();
            const bool isRMSEWithUncertainty = CatBoostOptions.LossFunctionDescription->GetLossFunction() == ELossFunction::RMSEWithUncertainty;
            if (isBoostFromAverage || isRMSEWithUncertainty) {
                CB_ENSURE(
                    !DataProvider->TargetData->GetBaseline()
                    && (!TestDataProvider || !TestDataProvider->TargetData->GetBaseline()),
                    "You can't use boost_from_average or RMSEWithUncertainty with baseline now.");
                state->StartingPoint = NCB::CalcOptimumConstApprox(
                    CatBoostOptions.LossFunctionDescription,
                    *DataProvider->TargetData->GetTarget(),
                    GetWeights(*DataProvider->TargetData));
            }

            const ui32 estimationPermutation = state->DataSets.PermutationsCount() - 1;
            const ui32 learnPermutationCount = estimationPermutation ? estimationPermutation
                                                                     : 1; //fallback to 1 permutation to learn and test
            state->PermutationFolds.resize(learnPermutationCount);

            state->Cursor.FoldData.resize(learnPermutationCount);

            const float start = state->StartingPoint ? (*state->StartingPoint)[0] : 0.0f;

            for (ui32 i = 0; i < learnPermutationCount; ++i) {
                auto& folds = state->PermutationFolds[i];
                const auto& permutation = state->DataSets.GetPermutation(i);
                TVector<float> baseline;
                if (DataProvider->MetaInfo.BaselineCount > 0) {
                    baseline = permutation.Gather((*DataProvider->TargetData->GetBaseline())[0]);
                } else {
                    baseline.resize(DataProvider->GetObjectCount(), start);
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
                const auto& permutation = state->DataSets.GetPermutation(estimationPermutation);
                TVector<float> baseline;
                if (DataProvider->MetaInfo.BaselineCount > 0) {
                    baseline = permutation.Gather((*DataProvider->TargetData->GetBaseline())[0]);
                } else {
                    baseline.resize(DataProvider->GetObjectCount(), start);
                }

                state->Cursor.Estimation = TMirrorBuffer<float>::CopyMapping(state->DataSets.GetDataSetForPermutation(estimationPermutation).GetTarget().GetTargets());
                state->Cursor.Estimation.Write(baseline);
            }

            if (TestDataProvider) {
                state->TestTarget = CreateTarget(state->DataSets.GetTestDataSet());
                state->TestCursor = TMirrorBuffer<float>::CopyMapping(state->DataSets.GetTestDataSet().GetTarget().GetTargets());
                if (TestDataProvider->MetaInfo.BaselineCount > 0) {
                    state->TestCursor.Write((*TestDataProvider->TargetData->GetBaseline())[0]);
                } else {
                    FillBuffer(state->TestCursor, start);
                }
            }

            return state;
        }

        THolder<TResultModel> Run() {
            CB_ENSURE(DataProvider);
            CB_ENSURE(ProgressTracker);

            auto state = CreateState();
            THolder<TResultModel> resultModel = MakeHolder<TResultModel>();

            if (ProgressTracker->NeedBestTestCursor()) {
                state->BestTestCursor = MakeHolder<TVec>();
                (*state->BestTestCursor).Reset(state->TestCursor.GetMapping());
            }
            ProgressTracker->MaybeRestoreFromSnapshot([&](IInputStream* in) {
                TDynamicBoostingProgress<TResultModel> progress;
                ::Load(in, progress);
                if (state->BestTestCursor) {
                    LoadCudaBuffer(in, state->BestTestCursor.Get());
                }
                WriteProgressToGpu(progress,
                                   FeaturesManager,
                                   *resultModel,
                                   state->Cursor,
                                   TestDataProvider ? &state->TestCursor : nullptr);
            });

            Fit(state->DataSets,
                state->Targets,
                state->PermutationFolds,
                state->TestTarget.Get(),
                ProgressTracker,
                &state->Cursor,
                TestDataProvider ? &state->TestCursor : nullptr,
                state->BestTestCursor.Get(),
                resultModel.Get());
            resultModel->SetBias(state->StartingPoint);
            return resultModel;
        }

        void RunModelBasedEval() {
            CB_ENSURE(false, "Model based eval is not supported for feature-parallel boosting.");
        }
    };
}
