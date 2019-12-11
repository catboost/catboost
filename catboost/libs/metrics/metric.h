#pragma once

#include "metric_holder.h"
#include "caching_metric.h"
#include "pfound.h"

#include <catboost/private/libs/data_types/pair.h>
#include <catboost/private/libs/data_types/query.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/private/libs/options/data_processing_options.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/private/libs/options/metric_options.h>
#include <catboost/libs/helpers/maybe_data.h>

#include <library/threading/local_executor/local_executor.h>
#include <library/containers/2d_array/2d_array.h>

#include <util/generic/fwd.h>
#include <util/generic/array_ref.h>

#include <cmath>

using NCatboostOptions::GetDefaultTargetBorder;

template <typename T>
struct TMetricParam {
    TMetricParam(const TString& name, const T& value, bool userDefined = false)
        : Name(name)
        , Value(value)
        , UserDefined(userDefined) {
    }

    explicit operator T() const {
        return Get();
    }

    T Get() const {
        Y_ASSERT(!IsIgnored());
        return Value;
    }

    TMetricParam<T>& operator =(const T& value) {
        Y_ASSERT(!IsIgnored());
        Value = value;
        UserDefined = true;
        return *this;
    }

    void SetDefaultValue(const T& value) {
        Y_ASSERT(!IsIgnored());
        Value = value;
        UserDefined = false;
    }

    bool IsUserDefined() const {
        return !IsIgnored() && UserDefined;
    }

    const TString& GetName() const {
        Y_ASSERT(!IsIgnored());
        return Name;
    }

    bool IsIgnored() const {
        return Ignored;
    }

    void MakeIgnored() {
        Ignored = true;
    }

private:
    TString Name;
    T Value = {};
    bool UserDefined = false;
    bool Ignored = false;
};

struct TCustomMetricDescriptor {
    using TEvalFuncPtr = TMetricHolder (*)(
        const TVector<TVector<double>>& approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        void* customData);
    using TGetDescriptionFuncPtr = TString (*)(void* customData);
    using TIsMaxOptimalFuncPtr = bool (*)(void* customData);
    using TGetFinalErrorFuncPtr = double (*)(const TMetricHolder& error, void* customData);

    void* CustomData = nullptr;
    TEvalFuncPtr EvalFunc = nullptr;
    TGetDescriptionFuncPtr GetDescriptionFunc = nullptr;
    TIsMaxOptimalFuncPtr IsMaxOptimalFunc = nullptr;
    TGetFinalErrorFuncPtr GetFinalErrorFunc = nullptr;
};

struct IMetric {
    virtual TMetricHolder Eval(
        const TVector<TVector<double>>& approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> queriesInfo,
        int begin,
        int end,
        NPar::TLocalExecutor& executor
    ) const = 0;
    virtual TMetricHolder Eval(
        const TVector<TVector<double>>& approx,
        const TVector<TVector<double>>& approxDelta,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> queriesInfo,
        int begin,
        int end,
        NPar::TLocalExecutor& executor
    ) const = 0;
    virtual TString GetDescription() const = 0;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const = 0;
    virtual EErrorType GetErrorType() const = 0;
    virtual double GetFinalError(const TMetricHolder& error) const = 0;
    virtual TVector<TString> GetStatDescriptions() const = 0;
    virtual bool IsAdditiveMetric() const = 0;
    virtual const TMap<TString, TString>& GetHints() const = 0;
    virtual void AddHint(const TString& key, const TString& value) = 0;
    virtual bool NeedTarget() const = 0;
    virtual ~IMetric() = default;

public:
    TMetricParam<bool> UseWeights{"use_weights", true};
};

struct TMetric: public IMetric {
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TVector<TString> GetStatDescriptions() const override;
    virtual const TMap<TString, TString>& GetHints() const override;
    virtual void AddHint(const TString& key, const TString& value) override;
    virtual bool NeedTarget() const override;
private:
    TMap<TString, TString> Hints;
};

struct TMultiRegressionMetric: public TMetric {
    virtual TMetricHolder Eval(
        TConstArrayRef<TVector<double>> approx,
        TConstArrayRef<TVector<double>> approxDelta,
        TConstArrayRef<TConstArrayRef<float>> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        NPar::TLocalExecutor& executor
    ) const = 0;
    TMetricHolder Eval(
        const TVector<TVector<double>>& /*approx*/,
        TConstArrayRef<float> /*target*/,
        TConstArrayRef<float> /*weight*/,
        TConstArrayRef<TQueryInfo> /*queriesInfo*/,
        int /*begin*/,
        int /*end*/,
        NPar::TLocalExecutor& /*executor*/
    ) const override {
        CB_ENSURE(false, "Multiregression metrics should not be used like regular metric");
    }
    TMetricHolder Eval(
        const TVector<TVector<double>>& /*approx*/,
        const TVector<TVector<double>>& /*approxDelta*/,
        bool /*isExpApprox*/,
        TConstArrayRef<float> /*target*/,
        TConstArrayRef<float> /*weight*/,
        TConstArrayRef<TQueryInfo> /*queriesInfo*/,
        int /*begin*/,
        int /*end*/,
        NPar::TLocalExecutor& /*executor*/
    ) const override {
        CB_ENSURE(false, "Multiregression metrics should not be used like regular metric");
    }
    EErrorType GetErrorType() const override final {
        return EErrorType::PerObjectError;
    }
};

static inline int GetMinBlockSize(int objectCount) {
    return 10000 < objectCount && objectCount < 100000 ? 1000 : 10000;
}

template <typename TEvalFunction>
static inline TMetricHolder ParallelEvalMetric(TEvalFunction eval, int minBlockSize, int begin, int end, NPar::TLocalExecutor& executor) {
    NPar::TLocalExecutor::TExecRangeParams blockParams(begin, end);

    const int threadCount = executor.GetThreadCount() + 1;
    const int effectiveBlockCount = Min(threadCount, (int)ceil((end - begin) * 1.0 / minBlockSize));

    blockParams.SetBlockCount(effectiveBlockCount);

    const int blockSize = blockParams.GetBlockSize();
    const ui32 blockCount = blockParams.GetBlockCount();

    TVector<TMetricHolder> results(blockCount);
    NPar::ParallelFor(executor, 0, blockCount, [&](int blockId) {
        const int from = begin + blockId * blockSize;
        const int to = Min<int>(begin + (blockId + 1) * blockSize, end);
        results[blockId] = eval(from, to);
    });

    TMetricHolder result;
    for (int i = 0; i < results.ysize(); ++i) {
        result.Add(results[i]);
    }
    return result;

}

template <typename TImpl>
struct TAdditiveMultiRegressionMetric: public TMultiRegressionMetric {
    TMetricHolder Eval(
        TConstArrayRef<TVector<double>> approx,
        TConstArrayRef<TVector<double>> approxDelta,
        TConstArrayRef<TConstArrayRef<float>> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        NPar::TLocalExecutor& executor
    ) const override {
        const auto evalMetric = [&](int from, int to) {
            return static_cast<const TImpl*>(this)->EvalSingleThread(
                approx, approxDelta, target, UseWeights.IsIgnored() || UseWeights ? weight : TVector<float>{}, from, to
            );
        };

        return ParallelEvalMetric(evalMetric, GetMinBlockSize(end - begin), begin, end, executor);
    }

    bool IsAdditiveMetric() const override final {
        return true;
    }
};

template <class TImpl>
struct TAdditiveMetric: public TMetric {
    TMetricHolder Eval(
        const TVector<TVector<double>>& approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> queriesInfo,
        int begin,
        int end,
        NPar::TLocalExecutor& executor
    ) const final {
        return Eval(approx, /*approxDelta*/{}, /*isExpApprox*/false, target, weight, queriesInfo, begin, end, executor);
    }

    TMetricHolder Eval(
        const TVector<TVector<double>>& approx,
        const TVector<TVector<double>>& approxDelta,
        bool isExpApprox,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        TConstArrayRef<TQueryInfo> queriesInfo,
        int begin,
        int end,
        NPar::TLocalExecutor& executor
    ) const final {
        const auto evalMetric = [&](int from, int to) {
            return static_cast<const TImpl*>(this)->EvalSingleThread(
                approx, approxDelta, isExpApprox, target, UseWeights.IsIgnored() || UseWeights ? weight : TVector<float>{}, queriesInfo, from, to
            );
        };

        return ParallelEvalMetric(evalMetric, GetMinBlockSize(end - begin), begin, end, executor);
    }

    bool IsAdditiveMetric() const final {
        return true;
    }
};

struct TNonAdditiveMetric: public TMetric {
    bool IsAdditiveMetric() const final {
        return false;
    }
};

THolder<IMetric> MakeCrossEntropyMetric(
    ELossFunction lossFunction,
    double border = GetDefaultTargetBorder());

THolder<IMetric> MakeCtrFactorMetric(double border = GetDefaultTargetBorder());

THolder<IMetric> MakeRMSEMetric();

THolder<IMetric> MakeLqMetric(double q);

THolder<IMetric> MakeR2Metric();

THolder<IMetric> MakeNumErrorsMetric(double k);

THolder<IMetric> MakeQuantileMetric(ELossFunction lossFunction, double alpha = 0.5, double delta = 1e-6);

THolder<IMetric> MakeExpectileMetric(ELossFunction lossFunction, double alpha = 0.5);

THolder<IMetric> MakeLogLinQuantileMetric(double alpha = 0.5);

THolder<IMetric> MakeMAPEMetric();

THolder<IMetric> MakePoissonMetric();

//Mean squared logarithmic error regression loss
THolder<IMetric> MakeMSLEMetric();

//Median absolute error regression loss
THolder<IMetric> MakeMedianAbsoluteErrorMetric();

//Symmetric mean absolute percentage error
THolder<IMetric> MakeSMAPEMetric();

//loglikelihood of prediction
THolder<IMetric> MakeLLPMetric();

THolder<IMetric> MakeMultiClassMetric();

THolder<IMetric> MakeMultiClassOneVsAllMetric();

THolder<IMetric> MakePairLogitMetric();

THolder<IMetric> MakeQueryRMSEMetric();

THolder<IMetric> MakeQueryCrossEntropyMetric(double alpha = 0.95);

THolder<IMetric> MakePFoundMetric(int topSize = -1, double decay = 0.85);

THolder<IMetric> MakeDcgMetric(int topSize = -1, ENdcgMetricType type = ENdcgMetricType::Base, bool normalized = true, ENdcgDenominatorType denominator = ENdcgDenominatorType::LogPosition);

THolder<IMetric> MakeQuerySoftMaxMetric();

THolder<IMetric> MakeFilteredDcgMetric(ENdcgMetricType type, ENdcgDenominatorType denominator);

THolder<IMetric> MakeBinClassAucMetric();
THolder<IMetric> MakeRankingAucMetric();
THolder<IMetric> MakeMultiClassAucMetric(int positiveClass);
THolder<IMetric> MakeMuAucMetric(const TMaybe<TVector<TVector<double>>>& misclassCostMatrix = Nothing());

THolder<IMetric> MakeBinClassPrecisionMetric(double border = GetDefaultTargetBorder());
THolder<IMetric> MakeMultiClassPrecisionMetric(int classesCount, int positiveClass);

THolder<IMetric> MakeBinClassRecallMetric(double border = GetDefaultTargetBorder());
THolder<IMetric> MakeMultiClassRecallMetric(int classesCount, int positiveClass);

THolder<IMetric> MakeBinClassBalancedAccuracyMetric(double border = GetDefaultTargetBorder());

THolder<IMetric> MakeBinClassBalancedErrorRate(double border = GetDefaultTargetBorder());

THolder<IMetric> MakeBinClassKappaMetric(double border = GetDefaultTargetBorder());
THolder<IMetric> MakeMultiClassKappaMetric(int classCount = 2);

THolder<IMetric> MakeBinClassWKappaMetric(double border = GetDefaultTargetBorder());
THolder<IMetric> MakeMultiClassWKappaMetric(int classCount = 2);

THolder<IMetric> MakeBinClassF1Metric(double border = GetDefaultTargetBorder());
THolder<IMetric> MakeMultiClassF1Metric(int classesCount, int positiveClass);

THolder<IMetric> MakeTotalF1Metric(int classesCount = 2);

THolder<IMetric> MakeMCCMetric(int classesCount = 2);

THolder<IMetric> MakeBrierScoreMetric();

THolder<IMetric> MakeHingeLossMetric();

THolder<IMetric> MakeHammingLossMetric(
    double border = GetDefaultTargetBorder(),
    bool isMulticlass = false);

THolder<IMetric> MakeZeroOneLossMetric(double border = GetDefaultTargetBorder());
THolder<IMetric> MakeZeroOneLossMetric(int classCount);

THolder<IMetric> MakePairAccuracyMetric();

THolder<IMetric> MakeMAPKMetric(int topSize = -1, double border = GetDefaultTargetBorder());

THolder<IMetric> MakeRecallAtKMetric(
    int topSize = -1,
    double border = GetDefaultTargetBorder());

THolder<IMetric> MakePrecisionAtKMetric(
    int topSize = -1,
    double border = GetDefaultTargetBorder());

THolder<IMetric> MakeCustomMetric(const TCustomMetricDescriptor& descriptor);

THolder<IMetric> MakeUserDefinedPerObjectMetric(const TMap<TString, TString>& params);

THolder<IMetric> MakeUserDefinedQuerywiseMetric(const TMap<TString, TString>& params);

THolder<IMetric> MakeAverageGainMetric(float topSize);

THolder<IMetric> MakeHuberLossMetric(double delta);

THolder<IMetric> MakeBinClassNormalizedGiniMetric(double border);
THolder<IMetric> MakeMultiClassNormalizedGiniMetric(int positiveClass);

THolder<IMetric> MakeFairLossMetric(double smoothness);

TVector<THolder<IMetric>> CreateMetricsFromDescription(const TVector<TString>& description, int approxDim);

TVector<THolder<IMetric>> CreateMetricFromDescription(const NCatboostOptions::TLossDescription& description, int approxDimension);

TVector<THolder<IMetric>> CreateMetrics(
    TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
    int approxDim);

TVector<THolder<IMetric>> CreateMetrics(
    const NCatboostOptions::TOption<NCatboostOptions::TMetricOptions>& evalMetricOptions,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    int approxDimension,
    bool hasWeights);

void InitializeEvalMetricIfNotSet(
    const NCatboostOptions::TOption<NCatboostOptions::TLossDescription>& objectiveMetricDescription,
    NCatboostOptions::TOption<NCatboostOptions::TLossDescription>* evalMetricDescription);

TVector<TString> GetMetricsDescription(const TVector<const IMetric*>& metrics);
TVector<TString> GetMetricsDescription(const TVector<THolder<IMetric>>& metrics);

TVector<bool> GetSkipMetricOnTrain(const TVector<const IMetric*>& metrics);
TVector<bool> GetSkipMetricOnTrain(const TVector<THolder<IMetric>>& metrics);

TVector<bool> GetSkipMetricOnTest(bool testHasTarget, const TVector<const IMetric*>& metrics);

TMetricHolder EvalErrors(
    const TVector<TVector<double>>& approx,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    const IMetric& error,
    NPar::TLocalExecutor* localExecutor
);

TMetricHolder EvalErrors(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    const IMetric& error,
    NPar::TLocalExecutor* localExecutor
);

TMetricHolder EvalErrors(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    const IMetric& error,
    NPar::TLocalExecutor* localExecutor
);

inline static TMetricHolder EvalErrors(
    const TVector<TVector<double>>& approx,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    const IMetric& error,
    NPar::TLocalExecutor* localExecutor
) {
    return EvalErrors(
        approx,
        /*approxDelta*/{},
        /*isExpApprox*/false,
        target,
        weight,
        queriesInfo,
        error,
        localExecutor
    );
}

inline bool IsMaxOptimal(const IMetric& metric) {
    EMetricBestValue bestValueType;
    float bestPossibleValue;
    metric.GetBestValue(&bestValueType, &bestPossibleValue);
    return bestValueType == EMetricBestValue::Max;
}

bool IsMaxOptimal(TStringBuf lossFunction);

bool IsMinOptimal(TStringBuf lossFunction);

void CheckPreprocessedTarget(
    TConstArrayRef<float> target,
    const NCatboostOptions::TLossDescription& lossDesciption,
    bool isNonEmptyAndNonConst,
    bool allowConstLabel
);

void CheckMetrics(const TVector<THolder<IMetric>>& metrics, const ELossFunction modelLoss);

bool IsQuantileLoss(const ELossFunction& loss);
