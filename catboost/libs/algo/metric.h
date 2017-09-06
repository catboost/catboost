#pragma once

#include "learn_context.h"
#include "error_holder.h"

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/hash.h>

struct IMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const = 0;
    virtual TString GetDescription() const = 0;
    virtual bool IsMaxOptimal() const = 0;
    virtual double GetFinalError(const TErrorHolder& error) const = 0;
    virtual ~IMetric() {
    }
};

struct TMetric: public IMetric {
    virtual double GetFinalError(const TErrorHolder& error) const override;
};

struct TLoglossMetric: public TMetric {
    explicit TLoglossMetric(ELossFunction lossFunction);
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    ELossFunction LossFunction;
};

struct TRMSEMetric: public TMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TErrorHolder& error) const override;
    virtual bool IsMaxOptimal() const override;
};

struct TR2Metric: public TMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual double GetFinalError(const TErrorHolder& error) const override;
    virtual bool IsMaxOptimal() const override;
};

class TQuantileMetric: public TMetric {
public:
    explicit TQuantileMetric(ELossFunction lossFunction, double alpha = 0.5);

    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    ELossFunction LossFunction;
    double Alpha;
};

class TLogLinearQuantileMetric: public TMetric {
public:
    explicit TLogLinearQuantileMetric(double alpha = 0.5);

    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    double Alpha;
};

struct TPoissonMetric: public TMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TMAPEMetric: public TMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TRocAUCMetric: public TMetric {
    TRocAUCMetric() = default;
    explicit TRocAUCMetric(int positiveClass);
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
};

struct TRecallMetric: public TMetric {
    TRecallMetric() = default;
    explicit TRecallMetric(int positiveClass);
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
};

struct TPrecisionMetric: public TMetric {
    TPrecisionMetric() = default;
    explicit TPrecisionMetric(int positiveClass);
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
};

struct TF1Metric: public TMetric {
    TF1Metric() = default;
    explicit TF1Metric(int positiveClass);
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;

private:
    int PositiveClass = 1;
    bool IsMultiClass = false;
};

struct TTotalF1Metric : public TMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TMCCMetric : public TMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TAccuracyMetric: public TMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TMulticlassLoglossMetric: public TMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

struct TMulticlassOneVsAllLoglossMetric : public TMetric {
    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
};

class TCustomMetric: public IMetric {
public:
    explicit TCustomMetric(const TCustomMetricDescriptor& descriptor);

    virtual TErrorHolder Eval(const yvector<yvector<double>>& approx,
                              const yvector<float>& target,
                              const yvector<float>& weight, int begin, int end,
                              NPar::TLocalExecutor& executor) const override;
    virtual TString GetDescription() const override;
    virtual bool IsMaxOptimal() const override;
    virtual double GetFinalError(const TErrorHolder& error) const override;

private:
    TCustomMetricDescriptor Descriptor;
};

yvector<THolder<IMetric>> CreateMetric(ELossFunction metric, const yhash<TString, TString>& params, int approxDimension);

yvector<THolder<IMetric>> CreateMetricFromDescription(const TString& description, int approxDimension);

yvector<THolder<IMetric>> CreateMetrics(const TFitParams& params, int approxDimension);
