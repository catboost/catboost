#pragma once

#include "learning_rate.h"

#include <catboost/libs/overfitting_detector/overfitting_detector.h>
#include <catboost/cuda/targets/target_base.h>
#include <catboost/cuda/targets/mse.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/gpu_data/fold_based_dataset.h>
#include <catboost/cuda/gpu_data/fold_based_dataset_builder.h>
#include <catboost/cuda/targets/target_options.h>

template <class TTarget,
          class TWeakModel>
class IBoostingListener {
public:
    using TConstVec = typename TTarget::TConstVec;

    virtual ~IBoostingListener() {
    }

    virtual void UpdateEnsemble(const TAdditiveModel<TWeakModel>& newEnsemble,
                                const TTarget& target,
                                const TConstVec& point) = 0;

    virtual void SetProgress(const TAdditiveModel<TWeakModel>& newEnsemble,
                             const TTarget& target,
                             const TConstVec& point) = 0;
};

template <class TTarget, class TWeakModel>
class TMetricLogger: public IBoostingListener<TTarget, TWeakModel> {
public:
    using TConstVec = typename TTarget::TConstVec;
    using TTargetStat = typename TMetricHelper<TTarget>::TTargetStat;

    TMetricLogger(const TString& messagePrefix,
                  TString outputPath = "")
        : MessagePrefix(messagePrefix)
        , OutputPath(outputPath)
    {
        if (OutputPath) {
            Out.Reset(new TOFStream(outputPath));
            (*Out) << "iter\t" << TTarget::TargetName() << Endl;
        }
    }

    void UpdateEnsemble(const TAdditiveModel<TWeakModel>& newEnsemble,
                        const TTarget& target,
                        const TConstVec& point) override {
        Y_UNUSED(newEnsemble);
        TMetricHelper<TTarget> metricHelper(target);
        metricHelper.SetPoint(point);
        if (BestEnsembleSize == 0 || metricHelper.IsBetter(BestStat)) {
            BestStat = metricHelper.GetStat();
            BestEnsembleSize = static_cast<ui32>(newEnsemble.Size());
        }

        MATRIXNET_INFO_LOG << MessagePrefix << metricHelper.ToTsv() << " best: " << metricHelper.Score(BestStat) << " (" << BestEnsembleSize << ")" << Endl;
        if (Out) {
            (*Out) << newEnsemble.Size() << "\t" << metricHelper.Score() << Endl;
        }
    }

    void SetProgress(const TAdditiveModel<TWeakModel>& model,
                     const TTarget& target,
                     const TConstVec& point) override {
        UpdateEnsemble(model, target, point);
    }

private:
    ui32 BestEnsembleSize = 0;
    TTargetStat BestStat;
    TString MessagePrefix;
    TString OutputPath;
    THolder<TOFStream> Out;
};

template <class TTarget,
          class TWeakModel>
class TIterationLogger: public IBoostingListener<TTarget, TWeakModel> {
public:
    using TConstVec = typename TTarget::TConstVec;

    void UpdateEnsemble(const TAdditiveModel<TWeakModel>& newEnsemble,
                        const TTarget& target,
                        const TConstVec& point) override {
        Y_UNUSED(newEnsemble);
        Y_UNUSED(target);
        Y_UNUSED(point);
        MATRIXNET_INFO_LOG << "Iteration #" << Iteration++ << " (ensemble size " << newEnsemble.Size() << ")" << Endl;
    }

    void SetProgress(const TAdditiveModel<TWeakModel>& model,
                     const TTarget& target,
                     const TConstVec& point) override {
        UpdateEnsemble(model, target, point);
    }

private:
    ui32 Iteration = 0;
};

class TBoostingOptions {
public:
    ui32 GetPermutationCount() const {
        return HasTimeFlag ? 1 : PermutationCount;
    }

    void SetPermutationCount(ui32 count)  {
        PermutationCount = count;
    }

    double GetGrowthRate() const {
        return GrowthRate;
    }

    bool DisableDontLookAhead() const {
        return DisableDontLookAheadFlag;
    }

    ui32 GetPermutationBlockSize() const {
        return PermutationBlockSize;
    }

    bool UseCpuRamForCatFeaturesDataSet() const {
        return UseCpuRamForCatFeaturesFlag;
    }

    TLearningRate GetLearningRate() const {
        return TLearningRate(Regularization);
    }

    bool IsCalcScores() const {
        return CalcScores;
    }
    ui32 GetIterationCount() const {
        return IterationCount;
    }

    ui32 GetMinFoldSize() const {
        return MinFoldSize;
    }

    const TString& GetLearnErrorLogPath() const {
        return LearnErrorLogPath;
    }

    const TString& GetTestErrorLogPath() const {
        return TestErrorLogPath;
    }

    double GetRandomStrength() const
    {
        return RandomStrength;
    }

    bool HasTime() const {
        return HasTimeFlag;
    }

    template <class TConfig>
    friend class TOptionsBinder;

private:
    ui32 PermutationCount = 4;
    bool HasTimeFlag = false;
    double GrowthRate = 2.0;
    bool DisableDontLookAheadFlag = false;
    ui32 PermutationBlockSize = 1;
    bool UseCpuRamForCatFeaturesFlag = false;
    ui32 IterationCount = 1000;
    ui32 MinFoldSize = 1024;
    double RandomStrength = 1.0;
    double Regularization = 0.5;
    bool CalcScores = true;
    TString LearnErrorLogPath;
    TString TestErrorLogPath;
};

template <template <class TMapping, class> class TTargetTemplate,
          class TWeakLearner,
          NCudaLib::EPtrType CatFeaturesStoragePtrType = NCudaLib::CudaDevice>
class TDontLookAheadBoosting {
public:
    using TTarget = TTargetTemplate<NCudaLib::TMirrorMapping, TDataSet<CatFeaturesStoragePtrType>>;
    using TResultModel = TAdditiveModel<typename TWeakLearner::TResultModel>;
    using TWeakModel = typename TWeakLearner::TResultModel;
    using TWeakModelStructure = typename TWeakLearner::TWeakModelStructure;
    using TVec = typename TTarget::TVec;
    using TConstVec = typename TTarget::TConstVec;
    using IListener = IBoostingListener<TTarget, TWeakModel>;

private:
    TBinarizedFeaturesManager& FeaturesManager;
    const TDataProvider* DataProvider;
    const TDataProvider* TestDataProvider;

    TRandom& Random;
    TWeakLearner& Weak;
    const TBoostingOptions& Config;
    const TTargetOptions& TargetOptions;

    yvector<IListener*> LearnListeners;
    yvector<IListener*> TestListeners;

    inline bool Stop(const ui32 iteration) {
        return iteration >= Config.GetIterationCount();
    }

private:
    struct TFold {
        TSlice EstimateSamples;
        TSlice QualityEvaluateSamples;
    };

    class TPermutationTarget {
    public:
        TPermutationTarget() = default;

        TPermutationTarget(yvector<THolder<TTarget>>&& targets)
            : Targets(std::move(targets))
        {
        }

        const TTarget& GetTarget(ui32 permutationId) const {
            return *Targets[permutationId];
        }

    private:
        yvector<THolder<TTarget>> Targets;
    };

    template <class TData>
    struct TFoldAndPermutationStorage {
        TFoldAndPermutationStorage() {
        }

        TFoldAndPermutationStorage(yvector<yvector<TData>>&& foldData,
                                   TData&& estimationData)
            : FoldData(std::move(foldData))
            , Estimation(std::move(estimationData))
        {
        }

        TData& Get(ui32 permutationId, ui32 foldId) {
            return FoldData.at(permutationId).at(foldId);
        }

        yvector<yvector<TData>> FoldData;
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
        ui32 suggestedBlockSize = Config.GetPermutationBlockSize();
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

    TDataSetsHolder<CatFeaturesStoragePtrType> CreateDataSet() const {
        CB_ENSURE(DataProvider);
        ui32 permutationBlockSize = GetPermutationBlockSize(DataProvider->GetSampleCount());

        TDataSetHoldersBuilder<CatFeaturesStoragePtrType> dataSetsHolderBuilder(FeaturesManager,
                                                                                *DataProvider,
                                                                                TestDataProvider,
                                                                                true,
                                                                                permutationBlockSize);

        return dataSetsHolderBuilder.BuildDataSet(Config.GetPermutationCount());
    }

    TPermutationTarget CreateTargets(const TDataSetsHolder<CatFeaturesStoragePtrType>& dataSets) const {
        yvector<THolder<TTarget>> targets;
        for (ui32 i = 0; i < dataSets.PermutationsCount(); ++i) {
            targets.push_back(CreateTarget(dataSets.GetDataSetForPermutation(i)));
        }
        return TPermutationTarget(std::move(targets));
    }

    THolder<TTarget> CreateTarget(const TDataSet<CatFeaturesStoragePtrType>& dataSet) const {
        return new TTarget(dataSet,
                           Random,
                           dataSet.GetTarget().GetObjectsSlice(),
                           TargetOptions);
    }

    inline ui32 MinEstimationSize(ui32 docCount) const {
        if (docCount < Config.GetMinFoldSize()) {
            return docCount / 2;
        }
        return Config.GetMinFoldSize();
    }

    yvector<TFold> CreateFolds(ui32 sampleCount,
                               double growthRate) const {
        const ui32 minEstimationSize = MinEstimationSize(sampleCount);
        CB_ENSURE(minEstimationSize, "Error: min learn size should be positive");
        CB_ENSURE(growthRate > 1.0, "Error: grow rate should be > 1.0");

        yvector<TFold> folds;
        if (Config.DisableDontLookAhead()) {
            folds.push_back({TSlice(0, sampleCount), TSlice(0, sampleCount)});
            return folds;
        }

        {
            const ui32 testEnd = Min(static_cast<ui32>(minEstimationSize * growthRate), sampleCount);
            folds.push_back({TSlice(0, minEstimationSize), TSlice(minEstimationSize, testEnd)});
        }

        while (folds.back().QualityEvaluateSamples.Right < sampleCount) {
            TSlice learnSlice = TSlice(0, folds.back().QualityEvaluateSamples.Right);
            const ui32 end = Min(static_cast<ui32>(folds.back().QualityEvaluateSamples.Right * growthRate), sampleCount);
            TSlice testSlice = TSlice(folds.back().QualityEvaluateSamples.Right,
                                      end);
            folds.push_back({learnSlice, testSlice});
        }
        return folds;
    }

    inline yvector<TFold> CreateFolds(const TTarget& target,
                                      const TDataSet<CatFeaturesStoragePtrType>& dataSet,
                                      double growthRate) const {
        //TODO: support query-based folds
        Y_UNUSED(target);
        return CreateFolds(static_cast<ui32>(dataSet.GetDataProvider().GetSampleCount()), growthRate);
    }

    using TCursor = TFoldAndPermutationStorage<TVec>;

    //don't look ahead boosting
    THolder<TResultModel> Fit(const TDataSetsHolder<CatFeaturesStoragePtrType>& dataSet,
                              const TPermutationTarget& target,
                              const ui32 offset,
                              const yvector<yvector<TFold>>& permutationFolds,
                              TCursor& cursor,
                              const TTarget* testTarget,
                              TVec* testCursor) {
        auto result = MakeHolder<TResultModel>();
        ui32 iteration = offset;
        auto& profiler = NCudaLib::GetProfiler();

        const ui32 permutationCount = dataSet.PermutationsCount();
        CB_ENSURE(permutationCount >= 1);
        const ui32 estimationPermutation = permutationCount - 1;
        const ui32 learnPermutationCount = estimationPermutation ? permutationCount - 1 : 1; //fallback

        auto learningRate = Config.GetLearningRate();

        auto startTimeBoosting = Now();

        while (!Stop(iteration)) {
            auto iterationTimeGuard = profiler.Profile("Boosting iteration");
            {
                {
                    auto startTime = Now();

                    //cache
                    THolder<TScopedCacheHolder> iterationCacheHolderPtr;
                    iterationCacheHolderPtr.Reset(new TScopedCacheHolder);
                    const double step = learningRate.Step(iteration);

                    auto weakModelStructure = [&]() -> TWeakModelStructure {
                        auto guard = profiler.Profile("Search for weak model structure");
                        const ui32 learnPermutationId =
                            learnPermutationCount > 1 ? static_cast<const ui32>(Random.NextUniformL() % (learnPermutationCount - 1))
                                                      : 0;

                        const auto& taskTarget = target.GetTarget(learnPermutationId);
                        const auto& taskDataSet = dataSet.GetDataSetForPermutation(learnPermutationId);
                        const auto& taskFolds = permutationFolds[learnPermutationId];

                        using TWeakTarget = TShiftedTargetSlice<TTarget>;

                        auto optimizer = Weak.template CreateStructureSearcher<TWeakTarget, TDataSet<CatFeaturesStoragePtrType>>(
                            *iterationCacheHolderPtr,
                            taskDataSet);

                        optimizer.SetRandomStrength(CalcScoreStDevMult(dataSet.GetDataProvider().GetSampleCount(), iteration * step));

                        if (Config.DisableDontLookAhead()) {
                            CB_ENSURE(taskFolds.size() == 1);
                            auto allSlice = taskTarget.GetIndices().GetObjectsSlice();
                            TShiftedTargetSlice<TTarget> shiftedTarget(taskTarget, allSlice, cursor.Get(learnPermutationId, 0).ConstCopyView());
                            optimizer.SetTarget(std::move(shiftedTarget));
                        } else {
                            for (ui32 foldId = 0; foldId < taskFolds.size(); ++foldId)
                            {
                                const auto& fold = taskFolds[foldId];


                                TShiftedTargetSlice<TTarget> learnTarget(taskTarget,
                                                                         fold.EstimateSamples,
                                                                         cursor.Get(learnPermutationId,
                                                                                    foldId).SliceView(fold.EstimateSamples));

                                TShiftedTargetSlice<TTarget> validateTarget(taskTarget,
                                                                            fold.QualityEvaluateSamples,
                                                                            cursor.Get(learnPermutationId,
                                                                                       foldId)
                                                                                    .SliceView(
                                                                                            fold.QualityEvaluateSamples));

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

                        auto estimator = Weak.template CreateEstimator<TTargetTemplate, TDataSet<CatFeaturesStoragePtrType>>(
                            weakModelStructure,
                            *iterationCacheHolderPtr);

                        for (ui32 permutation = 0; permutation < learnPermutationCount; ++permutation) {
                            auto& folds = permutationFolds[permutation];

                            for (ui32 foldId = 0; foldId < folds.size(); ++foldId) {
                                const auto& estimationSlice = folds[foldId].EstimateSamples;

                                estimator.AddEstimationTask(TargetSlice(target.GetTarget(permutation), estimationSlice),
                                                            cursor.Get(permutation, foldId).SliceView(estimationSlice),
                                                            &models.FoldData[permutation][foldId]);
                            }
                        }

                        if (!(Config.DisableDontLookAhead() && estimationPermutation == 0 /*no avereging permutation case*/)){
                            auto allSlice = dataSet.GetDataSetForPermutation(estimationPermutation).GetIndices().GetObjectsSlice();

                            estimator.AddEstimationTask(TargetSlice(target.GetTarget(estimationPermutation), allSlice),
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
                    if (Config.DisableDontLookAhead() && estimationPermutation == 0) {
                        models.Estimation = models.FoldData[0][0];
                    }
                    //
                    {
                        auto appendModelTime = profiler.Profile("Append models time");

                        auto addModelValue = Weak.template CreateAddModelValue<TDataSet<CatFeaturesStoragePtrType>>(
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

                    MATRIXNET_INFO_LOG << "Iteration time " << (Now() - startTime).SecondsFloat() << Endl;
                    result->AddWeakModel(models.Estimation);
                }

                {
                    auto learnListenerTimeGuard = profiler.Profile("Boosting learn listeners time: Learn");

                    for (auto& listener : LearnListeners) {
                        listener->UpdateEnsemble(*result,
                                                 target.GetTarget(estimationPermutation),
                                                 cursor.Estimation);
                    }
                }

                if (dataSet.HasTestDataSet()) {
                    auto testListenerTimeGuard = profiler.Profile("Boosting listeners time: Test");

                    for (auto& listener : TestListeners) {
                        listener->UpdateEnsemble(*result,
                                                 *testTarget,
                                                 *testCursor);
                    }
                }

                iteration++;
            }
            NCudaLib::GetCudaManager().DumpFreeMemory(TStringBuilder() << "Free gpu memory after iteration #" << iteration);
        }
        MATRIXNET_INFO_LOG << "Total time " << (Now() - startTimeBoosting).SecondsFloat() << Endl;

        return result;
    }

public:
    TDontLookAheadBoosting(TBinarizedFeaturesManager& binarizedFeaturesManager,
                           const TBoostingOptions& config,
                           const TTargetOptions& targetOptions,
                           TRandom& random,
                           TWeakLearner& weak)
        : FeaturesManager(binarizedFeaturesManager)
        , Random(random)
        , Weak(weak)
        , Config(config)
        , TargetOptions(targetOptions)
    {
    }

    virtual ~TDontLookAheadBoosting() {
    }

    TDontLookAheadBoosting& SetDataProvider(const TDataProvider& learnData,
                                            const TDataProvider* testData = nullptr) {
        DataProvider = &learnData;
        TestDataProvider = testData;
        return *this;
    }

    TDontLookAheadBoosting& RegisterLearnListener(IListener& listener) {
        LearnListeners.push_back(&listener);
        return *this;
    }

    TDontLookAheadBoosting& RegisterTestListener(IListener& listener) {
        Y_ENSURE(TestDataProvider, "Error: need test set for test listener");
        TestListeners.push_back(&listener);
        return *this;
    }

    TDontLookAheadBoosting& LoadProgress(TIStream& input) {
        Y_UNUSED(input);
        ythrow TCatboostException() << "unsupported yet";
        return *this;
    }

    struct TBoostingState {
        TDataSetsHolder<CatFeaturesStoragePtrType> DataSets;
        TPermutationTarget Targets;

        TCursor Cursor;

        TVec TestCursor;
        THolder<TTarget> TestTarget;

        yvector<yvector<TFold>> PermutationFolds;

        ui32 GetEstimationPermutation() const {
            return DataSets.PermutationsCount() - 1;
        }

        ui32 CurrentIteration = 0;
    };

    THolder<TBoostingState> CreateState() const {
        THolder<TBoostingState> state(new TBoostingState);
        state->DataSets = CreateDataSet();
        state->Targets = CreateTargets(state->DataSets);

        if (TestDataProvider) {
            state->TestTarget = CreateTarget(state->DataSets.GetTestDataSet());
            state->TestCursor = TMirrorBuffer<float>::CopyMapping(state->DataSets.GetTestDataSet().GetTarget());
            FillBuffer(state->TestCursor, 0.0f);
        }

        const ui32 estimationPermutation = state->DataSets.PermutationsCount() - 1;
        const ui32 learnPermutationCount = estimationPermutation ? estimationPermutation : 1; //fallback to 1 permutation to learn and test
        state->PermutationFolds.resize(learnPermutationCount);

        state->Cursor.FoldData.resize(learnPermutationCount);

        for (ui32 i = 0; i < learnPermutationCount; ++i) {
            auto& folds = state->PermutationFolds[i];

            folds = CreateFolds(state->Targets.GetTarget(i),
                                state->DataSets.GetDataSetForPermutation(i),
                                Config.GetGrowthRate());

            auto& foldCursors = state->Cursor.FoldData[i];
            foldCursors.resize(folds.size());

            for (ui32 fold = 0; fold < folds.size(); ++fold) {
                auto mapping = NCudaLib::TMirrorMapping(folds[fold].QualityEvaluateSamples.Right);
                foldCursors[fold] = TVec::Create(mapping);
                FillBuffer(foldCursors[fold], 0.0f);
            }
        }
        {
            state->Cursor.Estimation = TMirrorBuffer<float>::CopyMapping(state->DataSets.GetDataSetForPermutation(estimationPermutation).GetTarget());
            FillBuffer(state->Cursor.Estimation, 0.0f);
        }
        return state;
    }

    THolder<TResultModel> Run() {
        auto state = CreateState();
        return Fit(state->DataSets,
                   state->Targets,
                   state->CurrentIteration,
                   state->PermutationFolds,
                   state->Cursor,
                   state->TestTarget.Get(),
                   TestDataProvider ? &state->TestCursor : nullptr);
    }

    double CalcScoreStDevMult(const double sampleCount, double modelSize) {
        double modelExpLength = log(sampleCount);
        double modelLeft = exp(modelExpLength - modelSize);
        return Config.GetRandomStrength() * modelLeft / (1 + modelLeft);
    }

};
