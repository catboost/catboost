#pragma once

#include <util/string/cast.h>
#include <util/string/split.h>
#include <util/string/strip.h>

#include "histogram_points_and_bins.h"


class TExpressionVariable {
public:
    TExpressionVariable();
    TExpressionVariable(TString source);
    TExpressionVariable(double source);
    TExpressionVariable(const THistogramPointsAndBins& source);
    TExpressionVariable(const TExpressionVariable& source);
    explicit TExpressionVariable(bool source);

    TExpressionVariable& operator=(TString source);
    TExpressionVariable& operator=(double source);
    TExpressionVariable& operator=(THistogramPointsAndBins& source);
    TExpressionVariable& operator=(const TExpressionVariable& source);

    // covers std::string, TStringBuf, string literals
    template <typename T, typename = std::enable_if_t<std::is_constructible_v<TString, T>>>
    TExpressionVariable& operator=(T&& source) {
        return *this = TString{std::forward<T>(source)};
    }

    double Not();
    double Minus();
    double Min(const TExpressionVariable& secondOperand) const;
    double Max(const TExpressionVariable& secondOperand) const;
    double HistogramPercentile(const TExpressionVariable& percentile) const;
    double Or(const TExpressionVariable& secondOperand) const;
    double And(const TExpressionVariable& secondOperand) const;
    double Cond(const TExpressionVariable& secondOperand, const TExpressionVariable& u) const;
    TString StrCond(const TExpressionVariable& secondOperand, const TExpressionVariable& u) const;
    double Le(const TExpressionVariable& secondOperand) const;
    double L(const TExpressionVariable& secondOperand) const;
    double Ge(const TExpressionVariable& secondOperand) const;
    double G(const TExpressionVariable& secondOperand) const;
    double StrStartsWith(const TExpressionVariable& secondOperand) const;
    double StrLe(const TExpressionVariable& secondOperand) const;
    double StrL(const TExpressionVariable& secondOperand) const;
    double StrGe(const TExpressionVariable& secondOperand) const;
    double StrG(const TExpressionVariable& secondOperand) const;
    double VerComp(const TExpressionVariable& secondOperand, const double firstG, const double secondG) const;
    double VerE(const TExpressionVariable& secondOperand) const;
    double VerNe(const TExpressionVariable& secondOperand) const;
    double VerLe(const TExpressionVariable& secondOperand) const;
    double VerL(const TExpressionVariable& secondOperand) const;
    double VerGe(const TExpressionVariable& secondOperand) const;
    double VerG(const TExpressionVariable& secondOperand) const;
    double E(const TExpressionVariable& secondOperand) const;
    double Ne(const TExpressionVariable& secondOperand) const;
    double BitsOr(const TExpressionVariable& secondOperand) const;
    double BitsAnd(const TExpressionVariable& secondOperand) const;
    double Add(const TExpressionVariable& secondOperand) const;
    double Sub(const TExpressionVariable& secondOperand) const;
    double Mult(const TExpressionVariable& secondOperand) const;
    double Div(const TExpressionVariable& secondOperand) const;
    double Pow(const TExpressionVariable& secondOperand) const;
    double Exp() const;
    double Log() const;
    double Sqr() const;
    double Sqrt() const;
    double Sigmoid() const;

    double ToDouble() const;
    TString ToStr() const;
    THistogramPointsAndBins ToHistogramPointsAndBins() const;

private:
    bool IsZeroDoubleValue() const;
    bool IsEqual(const TExpressionVariable& secondOperand, const double eps) const;

    bool TryGetDoubleValue() const;
    bool TryParseDoubleVectorFromString(TVector<TString>& strVector, TVector<double>& doubleVector) const;
    bool TryParseFromStringToTHistogramPointsAndBins(THistogramPointsAndBins& pointsAndBins) const;

    static const double EPS;

    mutable bool HasStrValue;
    mutable bool HasDoubleValue;
    mutable bool HasHistogramPointsAndBinsValue;

    mutable TString StringValue;
    mutable double DoubleValue;
    mutable THistogramPointsAndBins HistogramPointsAndBinsValue;
};
