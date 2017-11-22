#pragma once

#include "metric_holder.h"
#include "ders_holder.h"

#include <catboost/libs/data/pair.h>

#include <library/threading/local_executor/local_executor.h>
#include <library/containers/2d_array/2d_array.h>

#include <util/generic/hash.h>

enum class ELossFunction {
    /* binary classification errors */

    Logloss,
    CrossEntropy,

    /* regression errors */

    RMSE,
    MAE,
    Quantile,
    LogLinQuantile,
    MAPE,
    Poisson,
    QueryRMSE,

    /* multiclassification errors */

    MultiClass,
    MultiClassOneVsAll,

    /* pair errors */

    PairLogit,

    /* regression metrics */

    R2,

    /* classification metrics */

    AUC,
    Accuracy,
    Precision,
    Recall,
    F1,
    TotalF1,
    MCC,

    /* pair metrics */

    PairAccuracy,

    /* custom errors */

    Custom
};

enum EErrorType {
    PerObjectError,
    PairwiseError,
    QuerywiseError
};

struct TCustomMetricDescriptor {
    void* CustomData;

    TMetricHolder (*EvalFunc)(const TVector<TVector<double>>& approx,
                             const TVector<float>& target,
                             const TVector<float>& weight,
                             int begin, int end, void* customData) = nullptr;
    TString (*GetDescriptionFunc)(void* customData) = nullptr;
    bool (*IsMaxOptimalFunc)(void* customData) = nullptr;
    double (*GetFinalErrorFunc)(const TMetricHolder& error, void* customData) = nullptr;
};

struct TCustomObjectiveDescriptor {
    void* CustomData;
    void (*CalcDersRange)(int count, const double* approxes, const float* targets,
                          const float* weights, TDer1Der2* ders, void* customData) = nullptr;
    void (*CalcDersMulti)(const TVector<double>& approx, float target, float weight,
                          TVector<double>* ders, TArray2D<double>* der2, void* customData) = nullptr;
};

struct IMetric {
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const = 0;

    virtual TMetricHolder EvalPairwise(const TVector<TVector<double>>& approx,
                                      const TVector<TPair>& pairs,
                                      int begin, int end) const = 0;

    virtual TMetricHolder EvalQuerywise(const TVector<TVector<double>>& approx,
                                       const TVector<float>& target,
                                       const TVector<float>& weight,
                                       const TVector<ui32>& queriesId,
                                       const THashMap<ui32, ui32>& queriesSize,
                                       int begin, int end) const = 0;

    virtual TString GetDescription() const = 0;
    virtual bool IsMaxOptimal() const = 0;
    virtual EErrorType GetErrorType() const = 0;
    virtual double GetFinalError(const TMetricHolder& error) const = 0;
    virtual bool IsAdditiveMetric() const = 0;
    virtual ~IMetric() {
    }
};

struct TMetric: public IMetric {
    virtual TMetricHolder EvalPairwise(const TVector<TVector<double>>& approx,
                                      const TVector<TPair>& pairs,
                                      int begin, int end) const override;
    virtual TMetricHolder EvalQuerywise(const TVector<TVector<double>>& approx,
                                       const TVector<float>& target,
                                       const TVector<float>& weight,
                                       const TVector<ui32>& queriesId,
                                       const THashMap<ui32, ui32>& queriesSize,
                                       int begin, int end) const override;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
};

struct TAdditiveMetric: public TMetric {
    bool IsAdditiveMetric() const final {
        return true;
    }
};

struct TNonAdditiveMetric: public TMetric {
    bool IsAdditiveMetric() const final {
        return false;
    }
};

struct TPairwiseMetric : public IMetric {
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TMetricHolder EvalQuerywise(const TVector<TVector<double>>& approx,
                                       const TVector<float>& target,
                                       const TVector<float>& weight,
                                       const TVector<ui32>& queriesId,
                                       const THashMap<ui32, ui32>& queriesSize,
                                       int begin, int end) const override;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
};

struct TPairwiseAdditiveMetric : public TPairwiseMetric {
    bool IsAdditiveMetric() const final {
        return true;
    }
};

struct TQuerywiseMetric : public IMetric {
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TMetricHolder EvalPairwise(const TVector<TVector<double>>& approx,
                                      const TVector<TPair>& pairs,
                                      int begin, int end) const override;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
};

struct TQuerywiseAdditiveMetric : public TQuerywiseMetric {
    bool IsAdditiveMetric() const final {
        return true;
    }
};

struct TCrossEntropyMetric: public TAdditiveMetric {
    explicit TCrossEntropyMetric(ELossFunction lossFunction);
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    ELossFunction LossFunction;
};

struct TRMSEMetric: public TAdditiveMetric {
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual bool IsMaxOptimal() const override;
};

class TQuantileMetric : public TAdditiveMetric {
public:
    explicit TQuantileMetric(ELossFunction lossFunction, double alpha = 0.5);

    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    ELossFunction LossFunction;
    double Alpha;
};

class TLogLinQuantileMetric : public TAdditiveMetric {
public:
    explicit TLogLinQuantileMetric(double alpha = 0.5);

    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    double Alpha;
};

struct TMAPEMetric : public TAdditiveMetric {
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TPoissonMetric : public TAdditiveMetric {
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TMultiClassMetric : public TAdditiveMetric {
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TMultiClassOneVsAllMetric : public TAdditiveMetric {
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TPairLogitMetric : public TPairwiseAdditiveMetric {
    virtual TMetricHolder EvalPairwise(const TVector<TVector<double>>& approx,
                                      const TVector<TPair>& pairs,
                                      int begin, int end) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TQueryRMSEMetric : public TQuerywiseAdditiveMetric {
    virtual TMetricHolder EvalQuerywise(const TVector<TVector<double>>& approx,
                                       const TVector<float>& target,
                                       const TVector<float>& weight,
                                       const TVector<ui32>& queriesId,
                                       const THashMap<ui32, ui32>& queriesSize,
                                       int begin, int end) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
private:
    double CalcQueryAvrg(int start, int count,
                         const TVector<double>& approxes,
                         const TVector<float>& targets,
                         const TVector<float>& weights) const;
};

struct TR2Metric: public TNonAdditiveMetric {
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    virtual bool IsMaxOptimal() const override;
};

struct TAUCMetric: public TNonAdditiveMetric {
    TAUCMetric() = default;
    explicit TAUCMetric(int positiveClass);
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
};

struct TAccuracyMetric : public TAdditiveMetric {
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TPrecisionMetric : public TNonAdditiveMetric {
    TPrecisionMetric() = default;
    explicit TPrecisionMetric(int positiveClass);
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
};

struct TRecallMetric: public TNonAdditiveMetric {
    TRecallMetric() = default;
    explicit TRecallMetric(int positiveClass);
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
};

struct TF1Metric: public TNonAdditiveMetric {
    TF1Metric() = default;
    explicit TF1Metric(int positiveClass);
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
};

struct TTotalF1Metric : public TNonAdditiveMetric {
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TMCCMetric : public TNonAdditiveMetric {
    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TPairAccuracyMetric : public TPairwiseAdditiveMetric {
    virtual TMetricHolder EvalPairwise(const TVector<TVector<double>>& approx,
                                      const TVector<TPair>& pairs,
                                      int begin, int end) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

class TCustomMetric: public IMetric {
public:
    explicit TCustomMetric(const TCustomMetricDescriptor& descriptor);

    virtual TMetricHolder Eval(const TVector<TVector<double>>& approx,
                              const TVector<float>& target,
                              const TVector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;

    virtual TMetricHolder EvalPairwise(const TVector<TVector<double>>& approx,
                                      const TVector<TPair>& pairs,
                                      int begin, int end) const override;

    virtual TMetricHolder EvalQuerywise(const TVector<TVector<double>>& approx,
                                       const TVector<float>& target,
                                       const TVector<float>& weight,
                                       const TVector<ui32>& queriesId,
                                       const THashMap<ui32, ui32>& queriesSize,
                                       int begin, int end) const override;

    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TMetricHolder& error) const override;
    //we don't now anything about custom metrics
    bool IsAdditiveMetric() const final {
        return false;
    }
private:
    TCustomMetricDescriptor Descriptor;
};



TVector<THolder<IMetric>> CreateMetric(ELossFunction metric, const THashMap<TString, TString>& params, int approxDimension);

TVector<THolder<IMetric>> CreateMetricFromDescription(const TString& description, int approxDimension);

TVector<THolder<IMetric>> CreateMetrics(const TMaybe<TString>& evalMetric, const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
                                        const TVector<TString>& customLoss, int approxDimension);

ELossFunction GetLossType(const TString& lossDescription);

THashMap<TString, float> GetLossParams(const TString& lossDescription);

