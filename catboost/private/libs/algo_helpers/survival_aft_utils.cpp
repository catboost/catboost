#include "survival_aft_utils.h"


EDistributionType DistributionFromString(const TString& distribution){
    if (distribution == "Normal"){
        return EDistributionType::Normal;
    }
    else if (distribution == "Logistic")
    {
        return EDistributionType::Logistic;
    }
    else
    {
        return EDistributionType::Extreme;
    }
}

double TNormalDistribution::CalcPdf(double x) const
{
    auto prec = -Sqr(x) / 2.0;
    FastExpInplace(&prec, 1);
    return prec * INV_SQRT_2PI;
}

double TNormalDistribution::CalcPdfDer1(double pdf, double x) const
{
    return -x * pdf;
}

double TNormalDistribution::CalcPdfDer2(double pdf, double x) const 
{
    return (std::pow(x, 2) - 1) * pdf;
}

double TNormalDistribution::ErrorFunction(const double x) const
{
    double coeffs[] = {
        -1.26551223,
        1.00002368,
        0.37409196,
        0.09678418,
        -0.18628806,
        0.27886807,
        -1.13520398,
        1.48851587,
        -0.82215223,
        0.17087277};

    double t = 1.0 / (1.0 + 0.5 * abs(x));
    double sum = -x * x;
    double powT = 1.0;
    for (double coef : coeffs)
    {
        sum += coef * powT;
        powT *= t;
    }
    FastExpInplace(&sum, 1);
    double tau = t * sum;
    if (x > 0)
    {
        return 1.0 - tau;
    }
    else
    {
        return tau - 1.0;
    }
}

double TNormalDistribution::CalcCdf (double x) const
{
    return 0.5 + 0.5 * ErrorFunction(x / sqrt(2.0));
}

EDistributionType TNormalDistribution::GetDistributionType() const{ 
    return EDistributionType::Normal;
}

double TExtremeDistribution::CalcPdf(double x) const
{   
    const double w = fast_exp(x);
    return !IsFinite(w) ? 0.0 : (w * fast_exp(-w));
}

double TExtremeDistribution::CalcPdfDer1(double pdf, double x) const
{
    const double w = fast_exp(x);
    return !IsFinite(w) ? 0.0 : ((1 - w) * pdf);
}

double TExtremeDistribution::CalcPdfDer2(double pdf, double x) const 
{
    const double w = fast_exp(x);
    if (!IsFinite(w) || !IsFinite(w * w)) {
      return 0.0;
    } else {
      return (std::pow(w,2) - 3 * w + 1) * pdf;
    }
}

double TExtremeDistribution::CalcCdf (double x) const
{
    const double w = fast_exp(x);
    return 1 - fast_exp(-w);
}

EDistributionType TExtremeDistribution::GetDistributionType() const{ 
    return EDistributionType::Extreme;
}

double TLogisticDistribution::CalcPdf(double x) const
{   
    const double w = fast_exp(x);
    const double sqrt_denominator = 1 + w;
    if (!IsFinite(w) || !IsFinite(std::pow(w,2))) {
      return 0.0;
    } else {
      return w / (std::pow(sqrt_denominator,2));
    }
}

double TLogisticDistribution::CalcPdfDer1(double pdf, double x) const
{
    const double w = exp(x);
    return !IsFinite(w) ? 0.0 : (pdf * (1 - w) / (1 + w));
}

double TLogisticDistribution::CalcPdfDer2(double pdf, double x) const 
{
    const double w = exp(x);
    const double w_squared = std::pow(w,2);
    if (!IsFinite(w) || !IsFinite(w_squared)) {
      return 0.0;
    } else {
      return pdf * (w_squared - 4 * w + 1) / ((1 + w) * (1 + w));
    }
}

double TLogisticDistribution::CalcCdf (double x) const
{
    const double w = exp(x);
    return !IsFinite(w) ? 1.0 : (w / (1 + w));
}

EDistributionType TLogisticDistribution::GetDistributionType() const{ 
    return EDistributionType::Logistic;
}

double InverseMonotoneTransform(double approx, double target, double scale)
{
    return (FastLogf(target) - approx) / scale;
}

double ClipDerivatives(double der, double minDerivative, double maxDerivative){
    return Max(Min(der,maxDerivative), minDerivative);
}

template<>
std::tuple<double, double> GetDerivativeLimits<EDistributionType::Normal>(EDerivativeOrder order, ECensoredType censoredType, double scale){
    switch (order){
        case EDerivativeOrder::First:
            switch (censoredType){
                case ECensoredType::IntervalCensored:
                case ECensoredType::Uncensored:
                    return std::make_tuple(MIN_FIRST_DER,MAX_FIRST_DER);
                case ECensoredType::RightCensored:
                    return std::make_tuple(MIN_FIRST_DER,0);
                case ECensoredType::LeftCensored:
                    return std::make_tuple(0,MAX_FIRST_DER);
            }
        case EDerivativeOrder::Second:
            switch (censoredType){
                case ECensoredType::IntervalCensored:
                case ECensoredType::Uncensored:
                    return std::make_tuple(1/std::pow(scale,2),1/std::pow(scale,2));
                case ECensoredType::RightCensored:
                    return std::make_tuple(1/std::pow(scale,2),MIN_SECOND_DER);
                case ECensoredType::LeftCensored:
                    return std::make_tuple(MIN_SECOND_DER,1/std::pow(scale,2));
            }
    }
}


template<>
std::tuple<double, double> GetDerivativeLimits<EDistributionType::Extreme>(EDerivativeOrder order, ECensoredType censoredType, double scale){
    switch (order){
        case EDerivativeOrder::First:
            switch (censoredType){
                case ECensoredType::IntervalCensored:
                case ECensoredType::Uncensored:
                    return std::make_tuple(-15,1/scale);
                case ECensoredType::RightCensored:
                    return std::make_tuple(-15,0);
                case ECensoredType::LeftCensored:
                    return std::make_tuple(0,1/scale);
            }
        case EDerivativeOrder::Second:
            switch (censoredType){
                case ECensoredType::IntervalCensored:
                case ECensoredType::Uncensored:
                case ECensoredType::RightCensored:
                    return std::make_tuple(15,MIN_SECOND_DER);
                case ECensoredType::LeftCensored:
                    return std::make_tuple(MIN_SECOND_DER,MIN_SECOND_DER);
            }
    }
}

template<>
std::tuple<double, double> GetDerivativeLimits<EDistributionType::Logistic>(EDerivativeOrder order, ECensoredType censoredType, double scale){
    switch (order){
        case EDerivativeOrder::First:
            switch (censoredType){
                case ECensoredType::IntervalCensored:
                case ECensoredType::Uncensored:
                    return std::make_tuple(-1/scale,1/scale);
                case ECensoredType::RightCensored:
                    return std::make_tuple(-1/scale,0);
                case ECensoredType::LeftCensored:
                    return std::make_tuple(0,1/scale);
            }
        case EDerivativeOrder::Second:
            switch (censoredType){
                case ECensoredType::IntervalCensored:
                case ECensoredType::Uncensored:
                case ECensoredType::RightCensored:
                case ECensoredType::LeftCensored:
                    return std::make_tuple(MIN_SECOND_DER,MIN_SECOND_DER);
            }
    }
}

std::tuple<double,double> DispatchDerivativeLimits(EDistributionType type, EDerivativeOrder derivativeOrder, ECensoredType censoredType, double scale) {
    switch (type){
        case EDistributionType::Normal:
            return GetDerivativeLimits<EDistributionType::Normal>(derivativeOrder, censoredType, scale);
        case EDistributionType::Extreme:
            return GetDerivativeLimits<EDistributionType::Extreme>(derivativeOrder, censoredType, scale);
        case EDistributionType::Logistic:
            return GetDerivativeLimits<EDistributionType::Logistic>(derivativeOrder, censoredType, scale);
        }
}