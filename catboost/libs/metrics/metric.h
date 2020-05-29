#pragma once

#include "metric_holder.h"
#include "caching_metric.h"
#include "pfound.h"
#include "enums.h"

#include <catboost/private/libs/data_types/pair.h>
#include <catboost/private/libs/data_types/query.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/private/libs/options/data_processing_options.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/private/libs/options/metric_options.h>
#include <catboost/libs/helpers/maybe_data.h>

#include <library/cpp/threading/local_executor/local_executor.h>
#include <library/cpp/containers/2d_array/2d_array.h>

#include <util/generic/fwd.h>
#include <util/generic/array_ref.h>

#include <cmath>
#include <utility>

using NCatboostOptions::GetDefaultTargetBorder;
using NCatboostOptions::GetDefaultPredictionBorder;

struct TMetricConfig {
    explicit TMetricConfig(ELossFunction metric, TLossParams params,
                           int approxDimension, TSet<TString>* validParams)
        : metric(metric)
        , params(std::move(params))
        , approxDimension(approxDimension)
        , validParams(validParams) {}

    double GetPredictionBorderOrDefault() const {
        return NCatboostOptions::GetPredictionBorderOrDefault(params.GetParamsMap(), GetDefaultPredictionBorder());
    }

    const TMap<TString, TString>& GetParamsMap() const {
        return params.GetParamsMap();
    }

    ELossFunction metric;
    TLossParams params;
    const int approxDimension;
    TSet<TString>* validParams;
};

using MetricsFactory = std::function<TVector<THolder<IMetric>>(const TMetricConfig&)>;

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

    using TEvalMultiregressionFuncPtr = TMetricHolder (*)(
        TConstArrayRef<TVector<double>> approx,
        TConstArrayRef<TConstArrayRef<float>> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        void* customData);

    using TGetDescriptionFuncPtr = TString (*)(void* customData);
    using TIsMaxOptimalFuncPtr = bool (*)(void* customData);
    using TGetFinalErrorFuncPtr = double (*)(const TMetricHolder& error, void* customData);

    void* CustomData = nullptr;
    TMaybe<TEvalFuncPtr> EvalFunc;
    TMaybe<TEvalMultiregressionFuncPtr> EvalMultiregressionFunc;
    TGetDescriptionFuncPtr GetDescriptionFunc = nullptr;
    TIsMaxOptimalFuncPtr IsMaxOptimalFunc = nullptr;
    TGetFinalErrorFuncPtr GetFinalErrorFunc = nullptr;

    bool IsMultiregressionMetric() const {
        CB_ENSURE(EvalFunc.Defined() || EvalMultiregressionFunc.Defined(), "Any custom eval function must be defined");
        CB_ENSURE(EvalFunc.Empty() || EvalMultiregressionFunc.Empty(), "Only one custom eval function must be defined");
        return EvalMultiregressionFunc.Defined();
    }
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
        const TConstArrayRef<TConstArrayRef<double>> approx,
        const TConstArrayRef<TConstArrayRef<double>> approxDelta,
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
    explicit TMetric(ELossFunction lossFunction, TLossParams descriptionParams);
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TVector<TString> GetStatDescriptions() const override;
    virtual const TMap<TString, TString>& GetHints() const override;
    virtual void AddHint(const TString& key, const TString& value) override;
    virtual bool NeedTarget() const override;
    // The default implementation of metric description formatting.
    // It uses LossFunction and DescriptionParams, which is user-specified metric options,
    // and constructs a Metric:key1=value1;key2=value2 string from them.
    // UseWeights is included in the description if the weights have been specified.
    virtual TString GetDescription() const override;
private:
    TMap<TString, TString> Hints;
    const ELossFunction LossFunction;
    const TLossParams DescriptionParams;
};

struct TMultiRegressionMetric: public TMetric {
    explicit TMultiRegressionMetric(ELossFunction lossFunction, const TLossParams& descriptionParams)
        : TMetric(lossFunction, descriptionParams) {}
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
    ) const final {
        CB_ENSURE(false, "Multiregression metrics should not be used like regular metric");
    }
    TMetricHolder Eval(
        const TConstArrayRef<TConstArrayRef<double>> /*approx*/,
        const TConstArrayRef<TConstArrayRef<double>> /*approxDelta*/,
        bool /*isExpApprox*/,
        TConstArrayRef<float> /*target*/,
        TConstArrayRef<float> /*weight*/,
        TConstArrayRef<TQueryInfo> /*queriesInfo*/,
        int /*begin*/,
        int /*end*/,
        NPar::TLocalExecutor& /*executor*/
    ) const final {
        CB_ENSURE(false, "Multiregression metrics should not be used like regular metric");
    }
    EErrorType GetErrorType() const final {
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

THolder<IMetric> MakeCtrFactorMetric(const TLossParams& params);

THolder<IMetric> MakeBinClassAucMetric(const TLossParams& params);
THolder<IMetric> MakeMultiClassAucMetric(const TLossParams& params, int positiveClass);

THolder<IMetric> MakeBinClassPrecisionMetric(const TLossParams& params,
                                             double predictionBorder = GetDefaultPredictionBorder());
THolder<IMetric> MakeMultiClassPrecisionMetric(const TLossParams& params,
                                               int classesCount, int positiveClass);

THolder<IMetric> MakeBinClassRecallMetric(const TLossParams& params,
                                          double predictionBorder = GetDefaultPredictionBorder());
THolder<IMetric> MakeMultiClassRecallMetric(const TLossParams& params,
                                            int classesCount, int positiveClass);

THolder<IMetric> MakeBinClassF1Metric(const TLossParams& params,
                                      double predictionBorder = GetDefaultPredictionBorder());
THolder<IMetric> MakeMultiClassF1Metric(const TLossParams& params,
                                        int classesCount, int positiveClass);

THolder<IMetric> MakeTotalF1Metric(const TLossParams& params,
                                   int classesCount = 2, EF1AverageType averageType = EF1AverageType::Weighted);

THolder<IMetric> MakeMCCMetric(const TLossParams& params,
                               int classesCount = 2);

THolder<IMetric> MakeBrierScoreMetric(const TLossParams& params);

THolder<IMetric> MakeCustomMetric(const TCustomMetricDescriptor& descriptor);

THolder<IMetric> MakeMultiClassPRAUCMetric(const TLossParams& params, int positiveClass);
THolder<IMetric> MakeBinClassPRAUCMetric(const TLossParams& params);

TVector<THolder<IMetric>> CreateMetricsFromDescription(const TVector<TString>& description, int approxDim);

TVector<THolder<IMetric>> CreateMetricFromDescription(const NCatboostOptions::TLossDescription& description,
                                                      int approxDimension);

// For tests.
TVector<THolder<IMetric>> CreateMetric(ELossFunction metric, const TLossParams& params, int approxDimension);

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
    const TConstArrayRef<TConstArrayRef<double>> approx,
    const TConstArrayRef<TConstArrayRef<double>> approxDelta,
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


namespace NCB {

void AppendTemporaryMetricsVector(TVector<THolder<IMetric>>&& src, TVector<THolder<IMetric>>* dst);

template <class MetricType>
TVector<THolder<IMetric>> AsVector(THolder<MetricType>&& metric) {
    TVector<THolder<IMetric>> result;
    result.emplace_back(std::move(metric));
    return result;
}

} // namespace internal
