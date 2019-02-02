#pragma once

#include "metric_holder.h"
#include "pfound.h"

#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/data_types/query.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/libs/options/metric_options.h>

#include <library/threading/local_executor/local_executor.h>
#include <library/containers/2d_array/2d_array.h>

#include <util/generic/fwd.h>

#include <cmath>

constexpr double GetDefaultClassificationBorder() {
    return 0.5;
}

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
        NPar::TLocalExecutor::TExecRangeParams blockParams(begin, end);

        const int threadCount = executor.GetThreadCount() + 1;
        const int MinBlockSize = 10000;
        const int effectiveBlockCount = Min(threadCount, (int)ceil((end - begin) * 1.0 / MinBlockSize));

        blockParams.SetBlockCount(effectiveBlockCount);

        const int blockSize = blockParams.GetBlockSize();
        const ui32 blockCount = blockParams.GetBlockCount();

        TVector<TMetricHolder> results(blockCount);
        NPar::ParallelFor(executor, 0, blockCount, [&](int blockId) {
            const int from = begin + blockId * blockSize;
            const int to = Min<int>(begin + (blockId + 1) * blockSize, end);
            Y_ASSERT(from < to);
            if (UseWeights.IsIgnored() || UseWeights)
                results[blockId] = static_cast<const TImpl*>(this)->EvalSingleThread(approx, approxDelta, isExpApprox, target, weight, queriesInfo, from, to);
            else
                results[blockId] = static_cast<const TImpl*>(this)->EvalSingleThread(approx, approxDelta, isExpApprox, target, {}, queriesInfo, from, to);
        });

        TMetricHolder result;
        for (int i = 0; i < results.ysize(); ++i) {
            result.Add(results[i]);
        }
        return result;
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
    double border = GetDefaultClassificationBorder());

THolder<IMetric> MakeCtrFactorMetric(double border = GetDefaultClassificationBorder());

THolder<IMetric> MakeRMSEMetric();

THolder<IMetric> MakeLqMetric(double q);

THolder<IMetric> MakeR2Metric();

THolder<IMetric> MakeNumErrorsMetric(double k);

THolder<IMetric> MakeQuantileMetric(ELossFunction lossFunction, double alpha = 0.5);

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

THolder<IMetric> MakeNdcgMetric(int topSize = -1, ENdcgMetricType type = ENdcgMetricType::Base);

THolder<IMetric> MakeQuerySoftMaxMetric();

THolder<IMetric> MakeBinClassAucMetric(double border = GetDefaultClassificationBorder());
THolder<IMetric> MakeMultiClassAucMetric(int positiveClass);

THolder<IMetric> MakeAccuracyMetric(double border = GetDefaultClassificationBorder());

THolder<IMetric> MakeBinClassPrecisionMetric(double border = GetDefaultClassificationBorder());
THolder<IMetric> MakeMultiClassPrecisionMetric(int positiveClass);

THolder<IMetric> MakeBinClassRecallMetric(double border = GetDefaultClassificationBorder());
THolder<IMetric> MakeMultiClassRecallMetric(int positiveClass);

THolder<IMetric> MakeBinClassBalancedAccuracyMetric(double border = GetDefaultClassificationBorder());

THolder<IMetric> MakeBinClassBalancedErrorRate(double border = GetDefaultClassificationBorder());

THolder<IMetric> MakeBinClassKappaMetric(double border = GetDefaultClassificationBorder());
THolder<IMetric> MakeMultiClassKappaMetric(int classCount = 2);

THolder<IMetric> MakeBinClassWKappaMetric(double border = GetDefaultClassificationBorder());
THolder<IMetric> MakeMultiClassWKappaMetric(int classCount = 2);

THolder<IMetric> MakeBinClassF1Metric(double border = GetDefaultClassificationBorder());
THolder<IMetric> MakeMultiClassF1Metric(int positiveClass);

THolder<IMetric> MakeTotalF1Metric(int classesCount = 2);

THolder<IMetric> MakeMCCMetric(int classesCount = 2);

THolder<IMetric> MakeBrierScoreMetric();

THolder<IMetric> MakeHingeLossMetric();

THolder<IMetric> MakeHammingLossMetric(
    double border = GetDefaultClassificationBorder(),
    bool isMulticlass = false);

THolder<IMetric> MakeZeroOneLossMetric(
    double border = GetDefaultClassificationBorder(),
    bool isMultiClass = false);

THolder<IMetric> MakePairAccuracyMetric();

THolder<IMetric> MakeMAPKMetric(int topSize = -1, double border = GetDefaultClassificationBorder());

THolder<IMetric> MakeRecallAtKMetric(
    int topSize = -1,
    double border = GetDefaultClassificationBorder());

THolder<IMetric> MakePrecisionAtKMetric(
    int topSize = -1,
    double border = GetDefaultClassificationBorder());

THolder<IMetric> MakeCustomMetric(const TCustomMetricDescriptor& descriptor);

THolder<IMetric> MakeUserDefinedPerObjectMetric(const TMap<TString, TString>& params);

THolder<IMetric> MakeUserDefinedQuerywiseMetric(const TMap<TString, TString>& params);

THolder<IMetric> MakeAverageGainMetric(float topSize);

THolder<IMetric> MakeHuberLossMetric(double delta);

TVector<THolder<IMetric>> CreateMetricsFromDescription(const TVector<TString>& description, int approxDim);

TVector<THolder<IMetric>> CreateMetricFromDescription(const NCatboostOptions::TLossDescription& description, int approxDimension);

TVector<THolder<IMetric>> CreateMetrics(
    TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
    int approxDim);

TVector<THolder<IMetric>> CreateMetrics(
    const NCatboostOptions::TOption<NCatboostOptions::TLossDescription>& lossFunctionOption,
    const NCatboostOptions::TOption<NCatboostOptions::TMetricOptions>& evalMetricOptions,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    int approxDimension);

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
    const THolder<IMetric>& error,
    NPar::TLocalExecutor* localExecutor
);

TMetricHolder EvalErrors(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDelta,
    bool isExpApprox,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TQueryInfo> queriesInfo,
    const THolder<IMetric>& error,
    NPar::TLocalExecutor* localExecutor
);

inline bool IsMaxOptimal(const IMetric& metric) {
    EMetricBestValue bestValueType;
    float bestPossibleValue;
    metric.GetBestValue(&bestValueType, &bestPossibleValue);
    return bestValueType == EMetricBestValue::Max;
}

void CheckPreprocessedTarget(
    TConstArrayRef<float> target,
    const NCatboostOptions::TLossDescription& lossDesciption,
    bool isLearnData,
    bool allowConstLabel = false
);

void CheckMetrics(const TVector<THolder<IMetric>>& metrics, const ELossFunction modelLoss);
