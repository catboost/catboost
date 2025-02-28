#include <limits>

#include "expression_variable.h"


TExpressionVariable::TExpressionVariable()
    : HasStrValue(false)
    , HasDoubleValue(false)
    , HasHistogramPointsAndBinsValue(false)
{
}

TExpressionVariable::TExpressionVariable(TString source)
    : HasStrValue(true)
    , HasDoubleValue(false)
    , HasHistogramPointsAndBinsValue(false)
    , StringValue(std::move(source))
{
}

TExpressionVariable::TExpressionVariable(double source)
    : HasStrValue(false)
    , HasDoubleValue(true)
    , HasHistogramPointsAndBinsValue(false)
    , DoubleValue(source)
{
}

TExpressionVariable::TExpressionVariable(const THistogramPointsAndBins& source)
    : HasStrValue(false)
    , HasDoubleValue(false)
    , HasHistogramPointsAndBinsValue(true)
    , HistogramPointsAndBinsValue(source)
{
}

TExpressionVariable::TExpressionVariable(const TExpressionVariable& source)
    : HasStrValue(source.HasStrValue)
    , HasDoubleValue(source.HasDoubleValue)
    , HasHistogramPointsAndBinsValue(source.HasHistogramPointsAndBinsValue)
{
    if (HasStrValue) {
        StringValue = source.StringValue;
    }
    if (HasHistogramPointsAndBinsValue) {
        HistogramPointsAndBinsValue = source.HistogramPointsAndBinsValue;
    }
    if (HasDoubleValue) {
        DoubleValue = source.DoubleValue;
    }
}

TExpressionVariable::TExpressionVariable(bool source)
    : HasStrValue(false)
    , HasDoubleValue(true)
    , HasHistogramPointsAndBinsValue(false)
    , DoubleValue(source ? 1.0 : 0.0)
{
}

TExpressionVariable& TExpressionVariable::operator=(TString source) {
    HasStrValue = true;
    HasDoubleValue = false;
    HasHistogramPointsAndBinsValue = false;
    StringValue = std::move(source);
    return *this;
}

TExpressionVariable& TExpressionVariable::operator=(double source) {
    HasStrValue = false;
    HasDoubleValue = true;
    HasHistogramPointsAndBinsValue = false;
    DoubleValue = source;
    return *this;
}

TExpressionVariable& TExpressionVariable::operator=(THistogramPointsAndBins& source) {
    HasStrValue = false;
    HasDoubleValue = true;
    HasHistogramPointsAndBinsValue = true;
    HistogramPointsAndBinsValue = source;
    return *this;
}

TExpressionVariable& TExpressionVariable::operator=(const TExpressionVariable& source) {
    HasStrValue = source.HasStrValue;
    HasDoubleValue = source.HasDoubleValue;
    HasHistogramPointsAndBinsValue = source.HasHistogramPointsAndBinsValue;
    if (HasStrValue) {
        StringValue = source.StringValue;
    }
    if (HasHistogramPointsAndBinsValue) {
        HistogramPointsAndBinsValue = source.HistogramPointsAndBinsValue;
    }
    if (HasDoubleValue) {
        DoubleValue = source.DoubleValue;
    }
    return *this;
}

double TExpressionVariable::Not() {
    return IsZeroDoubleValue() ? 1.0 : 0.0;
}
double TExpressionVariable::Minus() {
    return -ToDouble();
}
double TExpressionVariable::Min(const TExpressionVariable& secondOperand) const {
    return Le(secondOperand) ? ToDouble() : secondOperand.ToDouble();
}
double TExpressionVariable::Max(const TExpressionVariable& secondOperand) const {
    return G(secondOperand) ? ToDouble() : secondOperand.ToDouble();
}
double TExpressionVariable::HistogramPercentile(const TExpressionVariable& percentile) const {
    if (!ToHistogramPointsAndBins().IsValidData(percentile.ToDouble())) {
        return 0;
    }

    const auto& result = ToHistogramPointsAndBins().FindBinAndPartion(percentile.ToDouble());
    const auto& binIndex = result.first;
    const auto& partion = result.second;

    // if percentile in [last point; +inf) return last point * 1.1
    if (static_cast<size_t>(binIndex) == ToHistogramPointsAndBins().GetBins().size() - 1) {
        return ToHistogramPointsAndBins().GetPoints().back() * 1.1;
    }

    auto leftBorderPoint = binIndex != 0 ? ToHistogramPointsAndBins().GetPoints()[binIndex - 1] : 0;
    auto rightBorderPoint = ToHistogramPointsAndBins().GetPoints()[binIndex];

    // if right border is max_int return left border * 1.1
    if (rightBorderPoint == std::numeric_limits<int>::max()) {
        return ToHistogramPointsAndBins().GetPoints()[ToHistogramPointsAndBins().GetPoints().size() - 2] * 1.1;
    }

    return leftBorderPoint + (rightBorderPoint - leftBorderPoint) * partion;
}
double TExpressionVariable::Or(const TExpressionVariable& secondOperand) const {
    return !(IsZeroDoubleValue() && secondOperand.IsZeroDoubleValue());
}
double TExpressionVariable::And(const TExpressionVariable& secondOperand) const {
    return !(IsZeroDoubleValue() || secondOperand.IsZeroDoubleValue());
}
double TExpressionVariable::Cond(const TExpressionVariable& secondOperand, const TExpressionVariable& u) const {
    return !IsZeroDoubleValue() ? secondOperand.ToDouble() : u.ToDouble();
}
double TExpressionVariable::E(const TExpressionVariable& secondOperand) const {
    return IsEqual(secondOperand, EPS) ? 1.0 : 0.0;
}
double TExpressionVariable::Ne(const TExpressionVariable& secondOperand) const {
    return IsEqual(secondOperand, EPS) ? 0.0 : 1.0;
}
double TExpressionVariable::Le(const TExpressionVariable& secondOperand) const {
    return ToDouble() <= secondOperand.ToDouble() + EPS ? 1.0 : 0.0;
}
double TExpressionVariable::L(const TExpressionVariable& secondOperand) const {
    return ToDouble() < secondOperand.ToDouble() - EPS ? 1.0 : 0.0;
}
double TExpressionVariable::Ge(const TExpressionVariable& secondOperand) const {
    return ToDouble() >= secondOperand.ToDouble() - EPS ? 1.0 : 0.0;
}
double TExpressionVariable::G(const TExpressionVariable& secondOperand) const {
    return ToDouble() > secondOperand.ToDouble() + EPS ? 1.0 : 0.0;
}
double TExpressionVariable::StrStartsWith(const TExpressionVariable& secondOperand) const {
    return StringValue.StartsWith(secondOperand.StringValue) ? 1.0 : 0.0;
}
double TExpressionVariable::StrLe(const TExpressionVariable& secondOperand) const {
    return StringValue <= secondOperand.StringValue ? 1.0 : 0.0;
}
double TExpressionVariable::StrL(const TExpressionVariable& secondOperand) const {
    return StringValue < secondOperand.StringValue ? 1.0 : 0.0;
}
double TExpressionVariable::StrGe(const TExpressionVariable& secondOperand) const {
    return StringValue >= secondOperand.StringValue ? 1.0 : 0.0;
}
double TExpressionVariable::StrG(const TExpressionVariable& secondOperand) const {
    return StringValue > secondOperand.StringValue ? 1.0 : 0.0;
}
TString TExpressionVariable::StrCond(const TExpressionVariable& secondOperand, const TExpressionVariable& u) const {
    return !IsZeroDoubleValue() ? secondOperand.ToStr() : u.ToStr();
}
double TExpressionVariable::VerComp(const TExpressionVariable& secondOperand, const double firstG, const double secondG) const {
    /* Сравнение версий по стандарту uatraits:
    1) Версию мы всегда считаем из первых четырех чисел, остальные игнорируем.
    Если меньше четырех, добавляем недостающие нули.
    2) Если в каком-то из операндов мусор, какие-то префиксы или постфиксы, то
    операторы <#, <=#, >#, >=# и ==# вернут false, !=# вернёт true.
    См. примеры в тестах */
    const auto ss_first = StringSplitter(StripString(StringValue)).Split('.').Take(4);
    const auto ss_second = StringSplitter(StripString(secondOperand.StringValue)).Split('.').Take(4);
    auto first = ss_first.begin(), second = ss_second.begin();
    ui32 firstValue, secondValue;

    while (first != ss_first.end() && second != ss_second.end()) {
        if (!TryFromString<ui32>(*first, firstValue) || !TryFromString<ui32>(*second, secondValue)) {
            return 0.0;
        }
        if (firstValue > secondValue) {
            return firstG;
        } else if (firstValue < secondValue) {
            return secondG;
        }
        ++first;
        ++second;
    }

    while (first != ss_first.end()) {
        if (!TryFromString<ui32>(*first, firstValue)) {
            return 0.0;
        }
        if (firstValue > 0) {
            return firstG;
        }
        ++first;
    }

    while (second != ss_second.end()) {
        if (!TryFromString<ui32>(*second, secondValue)) {
            return 0.0;
        }
        if (secondValue > 0) {
            return secondG;
        }
        ++second;
    }

    return 0.0;
}
double TExpressionVariable::VerE(const TExpressionVariable& secondOperand) const {
    /* Сравнение версий по стандарту uatraits:
    1) Версию мы всегда считаем из первых четырех чисел, остальные игнорируем.
    Если меньше четырех, добавляем недостающие нули.
    2) Если в каком-то из операндов мусор, какие-то префиксы или постфиксы, то
    операторы <#, <=#, >#, >=# и ==# вернут false, !=# вернёт true.
    См. примеры в тестах */

    const auto ss_first = StringSplitter(StripString(StringValue)).Split('.').Take(4);
    const auto ss_second = StringSplitter(StripString(secondOperand.StringValue)).Split('.').Take(4);
    auto first = ss_first.begin(), second = ss_second.begin();
    ui32 firstValue, secondValue;

    while (first != ss_first.end() && second != ss_second.end()) {
        if (!TryFromString<ui32>(*first, firstValue) || !TryFromString<ui32>(*second, secondValue) || firstValue != secondValue) {
            return 0.0;
        }
        ++first;
        ++second;
    }

    while (first != ss_first.end()) {
        if (!TryFromString<ui32>(*first, firstValue) || firstValue != 0){
            return 0.0;
        }
        ++first;
    }

    while (second != ss_second.end()) {
        if (!TryFromString<ui32>(*second, secondValue) || secondValue != 0){
            return 0.0;
        }
        ++second;
    }

    return 1.0;
}
double TExpressionVariable::VerNe(const TExpressionVariable& secondOperand) const {
    return VerE(secondOperand) == 1.0 ? 0.0 : 1.0;
}
double TExpressionVariable::VerLe(const TExpressionVariable& secondOperand) const {
    return (VerComp(secondOperand, 0.0, 1.0) || VerE(secondOperand)) ? 1.0 : 0.0;
}
double TExpressionVariable::VerL(const TExpressionVariable& secondOperand) const {
    return VerComp(secondOperand, 0.0, 1.0);
}
double TExpressionVariable::VerGe(const TExpressionVariable& secondOperand) const {
    return (VerComp(secondOperand, 1.0, 0.0) || VerE(secondOperand)) ? 1.0 : 0.0;
}
double TExpressionVariable::VerG(const TExpressionVariable& secondOperand) const {
    return VerComp(secondOperand, 1.0, 0.0);
}
double TExpressionVariable::BitsOr(const TExpressionVariable& secondOperand) const {
    return static_cast<size_t>(ToDouble()) | static_cast<size_t>(secondOperand.ToDouble());
}
double TExpressionVariable::BitsAnd(const TExpressionVariable& secondOperand) const {
    return static_cast<size_t>(ToDouble()) & static_cast<size_t>(secondOperand.ToDouble());
}
double TExpressionVariable::Add(const TExpressionVariable& secondOperand) const {
    return ToDouble() + secondOperand.ToDouble();
}
double TExpressionVariable::Sub(const TExpressionVariable& secondOperand) const {
    return ToDouble() - secondOperand.ToDouble();
}
double TExpressionVariable::Mult(const TExpressionVariable& secondOperand) const {
    return ToDouble() * secondOperand.ToDouble();
}
double TExpressionVariable::Div(const TExpressionVariable& secondOperand) const {
    double denominator = secondOperand.ToDouble();
    if (denominator == 0) {
        if (ToDouble() == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::numeric_limits<double>::infinity();
    }
    return ToDouble() / denominator;
}
double TExpressionVariable::Pow(const TExpressionVariable& secondOperand) const {
    double exponent = secondOperand.ToDouble();
    if (exponent == 0 && ToDouble() == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::pow(ToDouble(), exponent);
}
double TExpressionVariable::Exp() const {
    return exp(ToDouble());
}
double TExpressionVariable::Log() const {
    return log(ToDouble());
}
double TExpressionVariable::Sqr() const {
    return pow(ToDouble(), 2.);
}
double TExpressionVariable::Sqrt() const {
    return pow(ToDouble(), 0.5);
}
double TExpressionVariable::Sigmoid() const {
    return 1.0 / (1. + exp(-ToDouble()));
}

double TExpressionVariable::ToDouble() const {
    if (HasDoubleValue) {
        return DoubleValue;
    }
    if (HasStrValue && TryFromString<double>(StringValue, DoubleValue)) {
        HasDoubleValue = true;
        return DoubleValue;
    }
    return double();
}

TString TExpressionVariable::ToStr() const {
    if (HasStrValue) {
        return StringValue;
    }
    if (HasHistogramPointsAndBinsValue) {
        StringValue = ToString(HistogramPointsAndBinsValue);
        return StringValue;
    }
    if (HasDoubleValue) {
        StringValue = ToString(DoubleValue);
        return StringValue;
    }
    return "";
}

THistogramPointsAndBins TExpressionVariable::ToHistogramPointsAndBins() const {
    if (HasHistogramPointsAndBinsValue) {
        return HistogramPointsAndBinsValue;
    }
    if (HasStrValue && TryParseFromStringToTHistogramPointsAndBins(HistogramPointsAndBinsValue)) {
        HasHistogramPointsAndBinsValue = true;
        return HistogramPointsAndBinsValue;
    }
    return THistogramPointsAndBins();
}


bool TExpressionVariable::IsZeroDoubleValue() const {
    return ToDouble() == 0.0;
}

bool TExpressionVariable::TryGetDoubleValue() const {
    if (HasDoubleValue) {
        return true;
    }
    if (HasStrValue && TryFromString<double>(StringValue, DoubleValue)) {
        HasDoubleValue = true;
        return true;
    }
    return false;
}

bool TExpressionVariable::TryParseDoubleVectorFromString(TVector<TString>& strVector, TVector<double>& doubleVector) const {
    for (size_t i = 0; i < strVector.size(); i++) {
        if (!TryFromString<double>(strVector[i], doubleVector[i])) {
            return false;
        }
    }
    return true;
}

bool TExpressionVariable::TryParseFromStringToTHistogramPointsAndBins(THistogramPointsAndBins& pointsAndBins) const {
    auto strPointsAndBins = StringSplitter(StringValue).Split(';').ToList<TString>();
    if (strPointsAndBins.size() != 2) {
        return false;
    }

    auto strPoints = StringSplitter(strPointsAndBins[0]).Split(',').ToList<TString>();
    auto strBins = StringSplitter(strPointsAndBins[1]).Split(',').ToList<TString>();

    if (strPoints.back() != "" || strBins.back() != "" || strPoints.size() != (strBins.size() - 1)) {
        return false;
    }

    strPoints.pop_back();
    strBins.pop_back();
    TVector<double> points(strPoints.size());
    TVector<double> bins(strBins.size());

    if (TryParseDoubleVectorFromString(strPoints, points) && TryParseDoubleVectorFromString(strBins, bins)) {
        pointsAndBins.SetPointsAndBins(points, bins);
        return true;
    }

    return false;
}

bool TExpressionVariable::IsEqual(const TExpressionVariable& secondOperand, const double eps) const {
    if (TryGetDoubleValue() && secondOperand.TryGetDoubleValue()) {
        return fabs(DoubleValue - secondOperand.DoubleValue) < eps;
    }
    if (HasStrValue && secondOperand.HasStrValue) {
        return StringValue == secondOperand.StringValue;
    }
    if (HasHistogramPointsAndBinsValue && secondOperand.HasHistogramPointsAndBinsValue) {
        return HistogramPointsAndBinsValue.IsEqual(secondOperand.HistogramPointsAndBinsValue, eps);
    }
    return false;
}
