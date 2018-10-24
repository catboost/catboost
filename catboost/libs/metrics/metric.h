#pragma once

#include "metric_holder.h"
#include "ders_holder.h"
#include "pfound.h"

#include <catboost/libs/algo/hessian.h>
#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/data_types/query.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/libs/options/metric_options.h>

#include <library/threading/local_executor/local_executor.h>
#include <library/containers/2d_array/2d_array.h>

#include <util/generic/hash.h>
#include <util/string/cast.h>

#include <cmath>

inline constexpr double GetDefaultClassificationBorder() {
    return 0.5;
}

template <typename T>
struct TMetricParam {
    TMetricParam(const TString& name, const T& value, bool userDefined = false)
        : Name(name)
        , Value(value)
        , UserDefined(userDefined) {
    }

    operator T() const {
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
    TString Name = {};
    T Value = {};
    bool UserDefined = false;
    bool Ignored = false;
};

struct TCustomMetricDescriptor {
    void* CustomData;
    TMetricHolder (*EvalFunc)(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        int begin,
        int end,
        void* customData
    ) = nullptr;
    TString (*GetDescriptionFunc)(void* customData) = nullptr;
    bool (*IsMaxOptimalFunc)(void* customData) = nullptr;
    double (*GetFinalErrorFunc)(const TMetricHolder& error, void* customData) = nullptr;
};

struct TCustomObjectiveDescriptor {
    void* CustomData;
    void (*CalcDersRange)(
        int count,
        const double* approxes,
        const float* targets,
        const float* weights,
        TDers* ders,
        void* customData
    ) = nullptr;
    void (*CalcDersMulti)(
        const TVector<double>& approx,
        float target,
        float weight,
        TVector<double>* ders,
        THessianInfo* der2,
        void* customData
    ) = nullptr;
};

struct IMetric {
    virtual TMetricHolder Eval(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
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
    virtual ~IMetric()
    {
    }

public:
    TMetricParam<bool> UseWeights{"use_weights", true};
};

struct TMetric: public IMetric {
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TVector<TString> GetStatDescriptions() const override;
    virtual const TMap<TString, TString>& GetHints() const override;
    virtual void AddHint(const TString& key, const TString& value) override;
private:
    TMap<TString, TString> Hints;
};

template <class TImpl>
struct TAdditiveMetric: public TMetric {
    TMetricHolder Eval(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
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
                results[blockId] = static_cast<const TImpl*>(this)->EvalSingleThread(approx, target, weight, queriesInfo, from, to);
            else
                results[blockId] = static_cast<const TImpl*>(this)->EvalSingleThread(approx, target, {}, queriesInfo, from, to);
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

struct TCrossEntropyMetric: public TAdditiveMetric<TCrossEntropyMetric> {
    explicit TCrossEntropyMetric(ELossFunction lossFunction, double border = GetDefaultClassificationBorder());
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    ELossFunction LossFunction;
    double Border = GetDefaultClassificationBorder();
};

class TCtrFactorMetric : public TAdditiveMetric<TCtrFactorMetric> {
public:
    explicit TCtrFactorMetric(double border = GetDefaultClassificationBorder()) : Border(border) {}
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    double Border;
};

struct TRMSEMetric: public TAdditiveMetric<TRMSEMetric> {
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
};

struct TLqMetric: public TAdditiveMetric<TLqMetric> {
    explicit TLqMetric(double q)
    : Q(q) {
        CB_ENSURE(Q >= 1, "Lq metric is defined for q >= 1, got " << q);
    }

    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    double Q;
};

struct TR2Metric: public TAdditiveMetric<TR2Metric> {
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int queryStartIndex,
        int queryEndIndex
    ) const;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
};



struct TNumErrorsMetric: public TAdditiveMetric<TNumErrorsMetric> {
    explicit TNumErrorsMetric(double k)
    : GreaterThen(k) {
        CB_ENSURE(k > 0, "Error: NumErrors metric requires num_erros > 0 parameter, got " << k);
    }

    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int queryStartIndex,
            int queryEndIndex
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    double GreaterThen;
};



class TQuantileMetric : public TAdditiveMetric<TQuantileMetric> {
public:
    explicit TQuantileMetric(ELossFunction lossFunction, double alpha = 0.5);
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    ELossFunction LossFunction;
    double Alpha;
};

class TLogLinQuantileMetric : public TAdditiveMetric<TLogLinQuantileMetric> {
public:
    explicit TLogLinQuantileMetric(double alpha = 0.5);
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    double Alpha;
};

struct TMAPEMetric : public TAdditiveMetric<TMAPEMetric> {
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
};

struct TPoissonMetric : public TAdditiveMetric<TPoissonMetric> {
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
};

//Mean squared logarithmic error regression loss
struct TMSLEMetric : public TAdditiveMetric<TMSLEMetric> {
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end
    ) const;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
};

//Median absolute error regression loss
struct TMedianAbsoluteErrorMetric : public TNonAdditiveMetric {
    virtual TMetricHolder Eval(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end,
            NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    TMedianAbsoluteErrorMetric() {
        UseWeights.MakeIgnored();
    }
};

//Symmetric mean absolute percentage error
struct TSMAPEMetric : public TAdditiveMetric<TSMAPEMetric> {
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end
    ) const;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
};

//loglikelihood of prediction
struct TLLPMetric : public TAdditiveMetric<TLLPMetric> {
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end
    ) const;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    virtual TVector<TString> GetStatDescriptions() const override;
};

struct TMultiClassMetric : public TAdditiveMetric<TMultiClassMetric> {
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
};

struct TMultiClassOneVsAllMetric : public TAdditiveMetric<TMultiClassOneVsAllMetric> {
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
};

struct TPairLogitMetric : public TAdditiveMetric<TPairLogitMetric> {
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int queryStartIndex,
        int queryEndIndex
    ) const;
    virtual EErrorType GetErrorType() const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
};

struct TQueryRMSEMetric : public TAdditiveMetric<TQueryRMSEMetric> {
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int queryStartIndex,
        int queryEndIndex
    ) const;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    double CalcQueryAvrg(
        int start,
        int count,
        const TVector<double>& approxes,
        const TVector<float>& targets,
        const TVector<float>& weights
    ) const;
};

struct TQueryCrossEntropyMetric : public TAdditiveMetric<TQueryCrossEntropyMetric> {
    explicit TQueryCrossEntropyMetric(double alpha = 0.95);
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int queryStartIndex,
            int queryEndIndex
    ) const;
    virtual EErrorType GetErrorType() const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    void AddSingleQuery(const double* approxes,
                        const float* target,
                        const float* weight,
                        int querySize,
                        TMetricHolder* metricHolder) const;
private:
    double Alpha;
};

struct TPFoundMetric : public TAdditiveMetric<TPFoundMetric> {
    explicit TPFoundMetric(int topSize = -1, double decay = 0.85);
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int queryStartIndex,
        int queryEndIndex
    ) const;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    int TopSize;
    double Decay;
};

struct TNdcgMetric: public TAdditiveMetric<TNdcgMetric> {
    explicit TNdcgMetric(int topSize = -1, ENdcgMetricType type = ENdcgMetricType::Base);
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int queryStartIndex,
            int queryEndIndex
    ) const;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    int TopSize;
    ENdcgMetricType MetricType;
};

struct TQuerySoftMaxMetric : public TAdditiveMetric<TQuerySoftMaxMetric> {
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int queryStartIndex,
        int queryEndIndex
    ) const;
    virtual EErrorType GetErrorType() const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    TMetricHolder EvalSingleQuery(
        int start,
        int count,
        const TVector<double>& approxes,
        const TVector<float>& targets,
        const TVector<float>& weights,
        TVector<double>* softmax
    ) const;
};

struct TAUCMetric: public TNonAdditiveMetric {
    static THolder<TAUCMetric> CreateBinClassMetric(double border = GetDefaultClassificationBorder());
    static THolder<TAUCMetric> CreateMultiClassMetric(int positiveClass);
    virtual TMetricHolder Eval(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end,
        NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
    double Border = GetDefaultClassificationBorder();

    explicit TAUCMetric(double border = GetDefaultClassificationBorder())
            : Border(border) {
        UseWeights.SetDefaultValue(false);
    }
};

struct TAccuracyMetric : public TAdditiveMetric<TAccuracyMetric> {
    explicit TAccuracyMetric(double border = GetDefaultClassificationBorder())
        : Border(border)
    {
    }
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    double Border = GetDefaultClassificationBorder();
};

struct TPrecisionMetric : public TAdditiveMetric<TPrecisionMetric> {
    static THolder<TPrecisionMetric> CreateBinClassMetric(double border = GetDefaultClassificationBorder());
    static THolder<TPrecisionMetric> CreateMultiClassMetric(int positiveClass);
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
    double Border = GetDefaultClassificationBorder();

    explicit TPrecisionMetric(double border = GetDefaultClassificationBorder())
            : Border(border)
    {
    }
    explicit TPrecisionMetric(int positiveClass);
};

struct TRecallMetric: public TAdditiveMetric<TRecallMetric> {
    static THolder<TRecallMetric> CreateBinClassMetric(double border = GetDefaultClassificationBorder());
    static THolder<TRecallMetric> CreateMultiClassMetric(int positiveClass);

    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
    double Border = GetDefaultClassificationBorder();

    explicit TRecallMetric(double border = GetDefaultClassificationBorder())
            : Border(border)
    {
    }
    explicit TRecallMetric(int positiveClass);
};

struct TBalancedAccuracyMetric: public TAdditiveMetric<TBalancedAccuracyMetric> {
    static THolder<TBalancedAccuracyMetric> CreateBinClassMetric(double border = GetDefaultClassificationBorder());
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end
    ) const;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    int PositiveClass = 1;
    double Border = GetDefaultClassificationBorder();

    explicit TBalancedAccuracyMetric(double border = GetDefaultClassificationBorder())
            : Border(border)
    {
    }
};

struct TBalancedErrorRate: public TAdditiveMetric<TBalancedErrorRate> {
    static THolder<TBalancedErrorRate> CreateBinClassMetric(double border = GetDefaultClassificationBorder());
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end
    ) const;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    int PositiveClass = 1;
    double Border = GetDefaultClassificationBorder();

    explicit TBalancedErrorRate(double border = GetDefaultClassificationBorder())
            : Border(border)
    {
    }
};

struct TKappaMetric: public TAdditiveMetric<TKappaMetric> {
    static THolder<TKappaMetric> CreateBinClassMetric(double border = GetDefaultClassificationBorder());
    static THolder<TKappaMetric> CreateMultiClassMetric(int classCount = 2);
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end
    ) const;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    double Border = GetDefaultClassificationBorder();
    int ClassCount = 2;

    explicit TKappaMetric(int classCount = 2, double border = GetDefaultClassificationBorder())
        : Border(border)
        , ClassCount(classCount) {
        UseWeights.MakeIgnored();
    }
};

struct TWKappaMatric: public TAdditiveMetric<TWKappaMatric> {
    static THolder<TWKappaMatric> CreateBinClassMetric(double border = GetDefaultClassificationBorder());
    static THolder<TWKappaMatric> CreateMultiClassMetric(int classCount = 2);
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end
    ) const;

    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual void GetBestValue(EMetricBestValue *valueType, float *bestValue) const override;

private:
    double Border = GetDefaultClassificationBorder();
    int ClassCount;

    explicit TWKappaMatric(int classCount = 2, double border = GetDefaultClassificationBorder())
        : Border(border)
        , ClassCount(classCount) {
        UseWeights.MakeIgnored();
    }
};

struct TF1Metric: public TAdditiveMetric<TF1Metric> {
    static THolder<TF1Metric> CreateF1Multiclass(int positiveClass);
    static THolder<TF1Metric> CreateF1BinClass(double border = GetDefaultClassificationBorder());
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    virtual TVector<TString> GetStatDescriptions() const override;
private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
    double Border = GetDefaultClassificationBorder();
};


struct TFactorizedF1Metric: public TAdditiveMetric<TFactorizedF1Metric> {
    static THolder<TFactorizedF1Metric> CreateFactorizedF1Metric(const TVector<ui32>& classFactorization);
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end
    ) const;
    TString GetDescription() const override;
    double GetFinalError(const TMetricHolder& error) const override;
    void GetBestValue(EMetricBestValue* valueType, float*) const override {
        *valueType = EMetricBestValue::Max;
    }
    virtual TVector<TString> GetStatDescriptions() const override;
private:
    TFactorizedF1Metric(const TVector<ui32>& factorization);
private:
    TVector<ui32> ClassFactorization;
};

struct TTotalF1Metric : public TAdditiveMetric<TTotalF1Metric> {
    explicit TTotalF1Metric(int classesCount = 2) : ClassCount(classesCount) {}
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TVector<TString> GetStatDescriptions() const override;
private:
    int ClassCount;
};

struct TMCCMetric : public TAdditiveMetric<TMCCMetric> {
    explicit TMCCMetric(int classesCount = 2) : ClassesCount(classesCount) {}
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end
    ) const;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    virtual TVector<TString> GetStatDescriptions() const override;
private:
    int ClassesCount;
};

struct TBrierScoreMetric : public TAdditiveMetric<TBrierScoreMetric> {
    virtual TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    TBrierScoreMetric() {
        UseWeights.MakeIgnored();
    }
};

struct THingeLossMetric : public TAdditiveMetric<THingeLossMetric> {
    virtual TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
};

struct THammingLossMetric : public TAdditiveMetric<THammingLossMetric> {
    explicit THammingLossMetric(double border = GetDefaultClassificationBorder(), bool isMultiClass = false);
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
private:
    double Border = GetDefaultClassificationBorder();
    bool IsMultiClass = false;
};

struct TZeroOneLossMetric : public TAdditiveMetric<TZeroOneLossMetric> {
    explicit TZeroOneLossMetric(double border = GetDefaultClassificationBorder(), bool isMultiClass = false);
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int begin,
            int end
    ) const;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
private:
    double Border = GetDefaultClassificationBorder();
    bool IsMultiClass = false;
};

struct TPairAccuracyMetric : public TAdditiveMetric<TPairAccuracyMetric> {
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int queryStartIndex,
        int queryEndIndex
    ) const;
    virtual EErrorType GetErrorType() const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
};


struct TMAPKMetric: public TAdditiveMetric<TMAPKMetric> {
    explicit TMAPKMetric(int topSize = -1, double border = GetDefaultClassificationBorder());
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int queryStartIndex,
            int queryEndIndex
    ) const;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    int TopSize;
    double Border;
};

struct TRecallAtKMetric: public TAdditiveMetric<TRecallAtKMetric> {
    explicit TRecallAtKMetric(int topSize = -1, double border = GetDefaultClassificationBorder());
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int queryStartIndex,
            int queryEndIndex
    ) const;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    int TopSize;
    double Border;
};

struct TPrecisionAtKMetric: public TAdditiveMetric<TPrecisionAtKMetric> {
    explicit TPrecisionAtKMetric(int topSize = -1, double border = GetDefaultClassificationBorder());
    TMetricHolder EvalSingleThread(
            const TVector<TVector<double>>& approx,
            const TVector<float>& target,
            const TVector<float>& weight,
            const TVector<TQueryInfo>& queriesInfo,
            int queryStartIndex,
            int queryEndIndex
    ) const;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    int TopSize;
    double Border;
};

class TCustomMetric: public IMetric {
public:
    explicit TCustomMetric(const TCustomMetricDescriptor& descriptor);
    virtual TMetricHolder Eval(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end,
        NPar::TLocalExecutor& executor
    ) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual TVector<TString> GetStatDescriptions() const override;
    virtual const TMap<TString, TString>& GetHints() const override;
    virtual void AddHint(const TString& key, const TString& value) override;
    //we don't now anything about custom metrics
    bool IsAdditiveMetric() const final {
        return false;
    }
private:
    TCustomMetricDescriptor Descriptor;
    TMap<TString, TString> Hints;
};

class TUserDefinedPerObjectMetric : public TMetric {
public:
    explicit TUserDefinedPerObjectMetric(const TMap<TString, TString>& params);
    virtual TMetricHolder Eval(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int begin,
        int end,
        NPar::TLocalExecutor& executor
    ) const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
    bool IsAdditiveMetric() const final {
        return true;
    }
private:
    double Alpha;
};

class TUserDefinedQuerywiseMetric : public TAdditiveMetric<TUserDefinedQuerywiseMetric> {
public:
    explicit TUserDefinedQuerywiseMetric(const TMap<TString, TString>& params);
    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int queryStartIndex,
        int queryEndIndex
    ) const;
    virtual EErrorType GetErrorType() const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    double Alpha;
};

class TAverageGain : public TAdditiveMetric<TAverageGain> {
public:
    explicit TAverageGain(float topSize)
        : TopSize(topSize) {
        CB_ENSURE(topSize > 0, "top size for AverageGain should be greater than 0");
        CB_ENSURE(topSize == (int)topSize, "top size for AverageGain should be an integer value");
    }

    TMetricHolder EvalSingleThread(
        const TVector<TVector<double>>& approx,
        const TVector<float>& target,
        const TVector<float>& weight,
        const TVector<TQueryInfo>& queriesInfo,
        int queryStartIndex,
        int queryEndIndex
    ) const;
    virtual EErrorType GetErrorType() const override;
    virtual TString GetDescription() const override;
    virtual void GetBestValue(EMetricBestValue* valueType, float* bestValue) const override;
private:
    int TopSize;
};

TVector<THolder<IMetric>> CreateMetricsFromDescription(const TVector<TString>& description, int approxDim);

TVector<THolder<IMetric>> CreateMetricFromDescription(const NCatboostOptions::TLossDescription& description, int approxDimension);

TVector<THolder<IMetric>> CreateMetrics(
    const NCatboostOptions::TOption<NCatboostOptions::TLossDescription>& lossFunctionOption,
    const NCatboostOptions::TOption<NCatboostOptions::TMetricOptions>& evalMetricOptions,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    int approxDimension
);

TVector<TString> GetMetricsDescription(const TVector<const IMetric*>& metrics);
inline TVector<TString> GetMetricsDescription(const TVector<THolder<IMetric>>& metrics) {
    return GetMetricsDescription(GetConstPointers(metrics));
}

TVector<bool> GetSkipMetricOnTrain(const TVector<const IMetric*>& metrics);
inline TVector<bool> GetSkipMetricOnTrain(const TVector<THolder<IMetric>>& metrics) {
    return GetSkipMetricOnTrain(GetConstPointers(metrics));
}

TMetricHolder EvalErrors(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TQueryInfo>& queriesInfo,
    const THolder<IMetric>& error,
    NPar::TLocalExecutor* localExecutor
);

inline bool IsMaxOptimal(const IMetric& metric) {
    EMetricBestValue bestValueType;
    float bestPossibleValue;
    metric.GetBestValue(&bestValueType, &bestPossibleValue);
    return bestValueType == EMetricBestValue::Max;
}

inline void CheckTarget(const TVector<float>& target, ELossFunction lossFunction) {
    if (lossFunction == ELossFunction::CrossEntropy) {
        auto targetBounds = CalcMinMax(target);
        CB_ENSURE(targetBounds.Min >= 0, "Min target less than 0: " + ToString(targetBounds.Min));
        CB_ENSURE(targetBounds.Max <= 1, "Max target greater than 1: " + ToString(targetBounds.Max));
    }

    if (lossFunction == ELossFunction::QuerySoftMax) {
        float minTarget = *MinElement(target.begin(), target.end());
        CB_ENSURE(minTarget >= 0, "Min target less than 0: " + ToString(minTarget));
    }

    if (IsMultiClassMetric(lossFunction)) {
        CB_ENSURE(AllOf(target, [](float x) { return int(x) == x && x >= 0; }), "if loss-function is MultiClass then each target label should be nonnegative integer");
    }
}

inline void CheckMetric(const ELossFunction metric, const ELossFunction modelLoss);

void CheckMetrics(const TVector<THolder<IMetric>>& metrics, const ELossFunction modelLoss);
