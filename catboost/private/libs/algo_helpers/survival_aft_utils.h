#pragma once

#include <util/generic/ymath.h>
#include <util/generic/string.h>
#include <cmath>
#include <tuple>

#include <library/cpp/fast_log/fast_log.h>
#include <library/cpp/fast_exp/fast_exp.h>

static constexpr double INV_SQRT_2PI = 0.398942280401432677939946;
static constexpr double MIN_FIRST_DER = -15.0;
static constexpr double MAX_FIRST_DER = 15.0;
static constexpr double MIN_SECOND_DER = 1e-16;
static constexpr double MAX_SECOND_DER = 15.0;
static constexpr double EPS = 1e-12; 

enum class EDistributionType {
    Normal,
    Logistic,
    Extreme
};

enum class ECensoredType{
    Uncensored,
    IntervalCensored,
    RightCensored,
    LeftCensored
};

enum class EDerivativeOrder{
    First,
    Second
};

EDistributionType DistributionFromString(const TString& distribution);

class IDistribution
{
public:
    virtual double CalcPdf(double x) const = 0;
    virtual double CalcCdf(double x) const = 0;
    virtual double CalcPdfDer1(
        double pdf,
        double x) const = 0;
    virtual double CalcPdfDer2(
        double pdf,
        double x) const = 0;
    virtual EDistributionType GetDistributionType() const = 0;
    virtual ~IDistribution()= default;
};

class TNormalDistribution : public IDistribution{
public:
    virtual double CalcPdf(double x) const override;

    virtual double CalcPdfDer1(
        double pdf,
        double x) const override;

    virtual double CalcPdfDer2(
        double pdf,
        double x) const override;

    virtual double CalcCdf (double x) const override;

    virtual EDistributionType GetDistributionType() const override;

    double ErrorFunction(const double x) const;
};

class TExtremeDistribution : public IDistribution{
public:
    virtual double CalcPdf(double x) const override;

    virtual double CalcPdfDer1(
        double pdf,
        double x) const override;

    virtual double CalcPdfDer2(
        double pdf,
        double x) const override;

    virtual double CalcCdf (double x) const override;

    virtual EDistributionType GetDistributionType() const override;

    double ErrorFunction(const double x) const;
};

class TLogisticDistribution : public IDistribution{
public:
    virtual double CalcPdf(double x) const override;

    virtual double CalcPdfDer1(
        double pdf,
        double x) const override;

    virtual double CalcPdfDer2(
        double pdf,
        double x) const override;

    virtual double CalcCdf (double x) const override;

    virtual EDistributionType GetDistributionType() const override;

    double ErrorFunction(const double x) const;
};

double InverseMonotoneTransform(double approx, double target, double scale);

double ClipDerivatives(double der, double minDerivative, double maxDerivative);

template<EDistributionType Distribution>
std::tuple<double, double> GetDerivativeLimits(EDerivativeOrder order, ECensoredType censoredType, double scale);

std::tuple<double,double> DispatchDerivativeLimits(EDistributionType type, EDerivativeOrder derivativeOrder, ECensoredType censoredType, double scale);