#pragma once

#include "learn_context.h"
#include "error_holder.h"

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/hash.h>

enum EErrorType {
    PerObjectError,
    PairwiseError,
    QuerywiseError
};

struct IMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const = 0;

    virtual TErrorHolder EvalPairwise(const yvector<yvector<double>>& approx,
                                      const yvector<TPair>& pairs,
                                      int begin, int end) const = 0;

    virtual TErrorHolder EvalQuerywise(const yvector<yvector<double>>& approx,
                                       const yvector<float>& target,
                                       const yvector<float>& weight,
                                       const yvector<ui32>& queriesId,
                                       const yhash<ui32, ui32>& queriesSize,
                                       int begin, int end) const = 0;

    virtual TString GetDescription() const = 0;
    virtual bool IsMaxOptimal() const = 0;
    virtual EErrorType GetErrorType() const = 0;
    virtual double GetFinalError(const TErrorHolder& error) const = 0;
    virtual bool IsAdditiveMetric() const = 0;
    virtual ~IMetric() {
    }
};

struct TMetric: public IMetric {
    virtual TErrorHolder EvalPairwise(const yvector<yvector<double>>& approx,
                                      const yvector<TPair>& pairs,
                                      int begin, int end) const override;
    virtual TErrorHolder EvalQuerywise(const yvector<yvector<double>>& approx,
                                       const yvector<float>& target,
                                       const yvector<float>& weight,
                                       const yvector<ui32>& queriesId,
                                       const yhash<ui32, ui32>& queriesSize,
                                       int begin, int end) const override;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TErrorHolder& error) const override;
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
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TErrorHolder EvalQuerywise(const yvector<yvector<double>>& approx,
                                       const yvector<float>& target,
                                       const yvector<float>& weight,
                                       const yvector<ui32>& queriesId,
                                       const yhash<ui32, ui32>& queriesSize,
                                       int begin, int end) const override;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TErrorHolder& error) const override;
};

struct TPairwiseAdditiveMetric : public TPairwiseMetric {
    bool IsAdditiveMetric() const final {
        return true;
    }
};

struct TQuerywiseMetric : public IMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TErrorHolder EvalPairwise(const yvector<yvector<double>>& approx,
                                      const yvector<TPair>& pairs,
                                      int begin, int end) const override;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TErrorHolder& error) const override;
};

struct TQuerywiseAdditiveMetric : public TQuerywiseMetric {
    bool IsAdditiveMetric() const final {
        return true;
    }
};

struct TCrossEntropyMetric: public TAdditiveMetric {
    explicit TCrossEntropyMetric(ELossFunction lossFunction);
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    ELossFunction LossFunction;
};

struct TRMSEMetric: public TAdditiveMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TErrorHolder& error) const override;
    virtual bool IsMaxOptimal() const override;
};

class TQuantileMetric : public TAdditiveMetric {
public:
    explicit TQuantileMetric(ELossFunction lossFunction, double alpha = 0.5);

    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
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

    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    double Alpha;
};

struct TMAPEMetric : public TAdditiveMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TPoissonMetric : public TAdditiveMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TMultiClassMetric : public TAdditiveMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TMultiClassOneVsAllMetric : public TAdditiveMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TPairLogitMetric : public TPairwiseAdditiveMetric {
    virtual TErrorHolder EvalPairwise(const yvector<yvector<double>>& approx,
                                      const yvector<TPair>& pairs,
                                      int begin, int end) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TQueryRMSEMetric : public TQuerywiseAdditiveMetric {
    virtual TErrorHolder EvalQuerywise(const yvector<yvector<double>>& approx,
                                       const yvector<float>& target,
                                       const yvector<float>& weight,
                                       const yvector<ui32>& queriesId,
                                       const yhash<ui32, ui32>& queriesSize,
                                       int begin, int end) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
private:
    double CalcQueryAvrg(int start, int count,
                         const yvector<double>& approxes,
                         const yvector<float>& targets,
                         const yvector<float>& weights) const;
};

struct TR2Metric: public TNonAdditiveMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TErrorHolder& error) const override;
    virtual bool IsMaxOptimal() const override;
};

struct TAUCMetric: public TNonAdditiveMetric {
    TAUCMetric() = default;
    explicit TAUCMetric(int positiveClass);
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
};

struct TAccuracyMetric : public TAdditiveMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TPrecisionMetric : public TNonAdditiveMetric {
    TPrecisionMetric() = default;
    explicit TPrecisionMetric(int positiveClass);
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
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
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
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
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
};

struct TTotalF1Metric : public TNonAdditiveMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TMCCMetric : public TNonAdditiveMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TPairAccuracyMetric : public TPairwiseAdditiveMetric {
    virtual TErrorHolder EvalPairwise(const yvector<yvector<double>>& approx,
                                      const yvector<TPair>& pairs,
                                      int begin, int end) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

class TCustomMetric: public IMetric {
public:
    explicit TCustomMetric(const TCustomMetricDescriptor& descriptor);

    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;

    virtual TErrorHolder EvalPairwise(const yvector<yvector<double>>& approx,
                                      const yvector<TPair>& pairs,
                                      int begin, int end) const override;

    virtual TErrorHolder EvalQuerywise(const yvector<yvector<double>>& approx,
                                       const yvector<float>& target,
                                       const yvector<float>& weight,
                                       const yvector<ui32>& queriesId,
                                       const yhash<ui32, ui32>& queriesSize,
                                       int begin, int end) const override;

    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
    virtual EErrorType GetErrorType() const override;
    virtual double GetFinalError(const TErrorHolder& error) const override;
    //we don't now anything about custom metrics
    bool IsAdditiveMetric() const final {
        return false;
    }
private:
    TCustomMetricDescriptor Descriptor;
};

yvector<THolder<IMetric>> CreateMetric(ELossFunction metric, const yhash<TString, TString>& params, int approxDimension);

yvector<THolder<IMetric>> CreateMetricFromDescription(const TString& description, int approxDimension);

yvector<THolder<IMetric>> CreateMetrics(const TFitParams& params, int approxDimension);
