#pragma once

#include <catboost/libs/overfitting_detector/overfitting_detector.h>
#include <catboost/cuda/targets/target_base.h>
#include <catboost/cuda/targets/mse.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/gpu_data/fold_based_dataset.h>
#include <catboost/cuda/gpu_data/fold_based_dataset_builder.h>
#include <catboost/cuda/targets/target_options.h>
#include <util/stream/format.h>

template <class TTarget,
          class TWeakModel>
class IBoostingListener {
public:
    using TConstVec = typename TTarget::TConstVec;

    virtual ~IBoostingListener() {
    }

    virtual void Init(const TTarget& target,
                      const TConstVec& point) {
        Y_UNUSED(target);
        Y_UNUSED(point);
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

    explicit TMetricLogger(const TString& messagePrefix,
                           TString outputPath = "")
        : MessagePrefix(messagePrefix)
        , OutputPath(outputPath)
    {
        if (OutputPath) {
            Out.Reset(new TOFStream(outputPath));
            (*Out) << "iter\t" << TTarget::TargetName() << Endl;
        }
    }

    static TStringBuf GetMetricName() {
        return TTarget::TargetName();
    }

    static bool IsMinOptimal() {
        return TTarget::IsMinOptimal();
    }

    ui32 GetBestIteration() const {
        return BestEnsembleSize;
    }

    void RegisterOdDetector(IOverfittingDetector* odDetector) {
        OdDetector = odDetector;
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
        if (OdDetector) {
            OdDetector->AddError(metricHelper.Score());
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
    IOverfittingDetector* OdDetector = nullptr;
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

template <class TTarget,
          class TWeakModel>
class TTimeWriter: public IBoostingListener<TTarget, TWeakModel> {
public:
    using TConstVec = typename TTarget::TConstVec;

    TTimeWriter(const ui32 totalIterations,
                const TString& outputFile)
        : TotalIterations(totalIterations)
        , Output(outputFile)
        , StartTime(Now())
    {
    }

    void Init(const TTarget& target,
              const TConstVec& point) override {
        Y_UNUSED(target);
        Y_UNUSED(point);
        StartTime = Now();
    }

    void UpdateEnsemble(const TAdditiveModel<TWeakModel>& newEnsemble,
                        const TTarget& target,
                        const TConstVec& point) override {
        Y_UNUSED(target);
        Y_UNUSED(point);
        const ui32 passedIterations = newEnsemble.Size();

        auto passedTime = (Now() - StartTime).GetValue();
        auto currentIteration = (Now() - PrevIteration).GetValue();
        MATRIXNET_INFO_LOG << "Iteration time\tcurrent: " << HumanReadable(TDuration(currentIteration));
        MATRIXNET_INFO_LOG << "\ttotal: " << HumanReadable(TDuration(passedTime));
        auto remainingTime = passedTime * (TotalIterations - passedIterations) / (passedIterations - FirstIteration);
        if (newEnsemble.Size() != TotalIterations) {
            MATRIXNET_INFO_LOG << "\tremaining: " << HumanReadable(TDuration(remainingTime));
        }
        Output << newEnsemble.Size() - 1 << "\t" << TDuration(remainingTime).MilliSeconds() << "\t" << TDuration(passedTime).MilliSeconds() << Endl;
        MATRIXNET_INFO_LOG << Endl;
        PrevIteration = Now();
    }

    void SetProgress(const TAdditiveModel<TWeakModel>& model,
                     const TTarget& target,
                     const TConstVec& point) override {
        Y_UNUSED(target);
        Y_UNUSED(point);
        FirstIteration = model.Size();
    }

private:
    ui32 TotalIterations = 0;
    TOFStream Output;
    ui32 Iteration = 0;
    ui32 FirstIteration = 0;
    TInstant StartTime;
    TInstant PrevIteration;
};
