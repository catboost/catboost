#include "expression_variable.h"


TExpressionVariable::TExpressionVariable()
    : HasStrValue(true)
    , HasDoubleValue(true)
    , HasHistogramPointsAndBinsValue(false)
{
}

TExpressionVariable::TExpressionVariable(const TString& source)
    : HasStrValue(true)
    , HasDoubleValue(false)
    , HasHistogramPointsAndBinsValue(false)
    , StringValue(source)
{
}

TExpressionVariable::TExpressionVariable(double source)
    : HasStrValue(false)
    , HasDoubleValue(false)
    , HasHistogramPointsAndBinsValue(false)
    , DoubleValue(source)
{
}

TExpressionVariable::TExpressionVariable(const THistogramPointsAndBins& source)
    : HasStrValue(false)
    , HasDoubleValue(true)
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
    } else if (HasHistogramPointsAndBinsValue) {
        HistogramPointsAndBinsValue = source.HistogramPointsAndBinsValue;
    } else {
        DoubleValue = source.DoubleValue;
    }
}

TExpressionVariable::TExpressionVariable(bool source)
    : HasStrValue(false)
    , HasDoubleValue(false)
    , HasHistogramPointsAndBinsValue(false)
    , DoubleValue(source ? 1.0 : 0.0)
{
}

TExpressionVariable& TExpressionVariable::operator=(const TString& source) {
    HasStrValue = true;
    HasDoubleValue = false;
    HasHistogramPointsAndBinsValue = false;
    StringValue = source;
    return *this;
}

TExpressionVariable& TExpressionVariable::operator=(double source) {
    HasStrValue = false;
    HasDoubleValue = false;
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
    } else if (HasHistogramPointsAndBinsValue) {
        HistogramPointsAndBinsValue = source.HistogramPointsAndBinsValue;
    } else {
        DoubleValue = source.DoubleValue;
    }
    return *this;
}

double TExpressionVariable::Not() {
    return IsEmpty() ? 1.0 : 0.0;
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
    if (ToHistogramPointsAndBins().IsInvalidData(percentile.ToDouble())) {
        return 0;
    }

    const auto& result = ToHistogramPointsAndBins().FindBinAndPartion(percentile.ToDouble());
    const auto& binIndex = result.first;
    const auto& partion = result.second;

    // [last point; +inf)
    if (static_cast<size_t>(binIndex) == ToHistogramPointsAndBins().GetBins().size() - 1) {
        return ToHistogramPointsAndBins().GetPoints().back() * 1.1;
    }

    // (-inf; first point] or exactly in points
    if (binIndex == 0 || partion == 0) {
        return ToHistogramPointsAndBins().GetPoints()[binIndex];
    }
    return ToHistogramPointsAndBins().GetPoints()[binIndex - 1] + (ToHistogramPointsAndBins().GetPoints()[binIndex] - ToHistogramPointsAndBins().GetPoints()[binIndex - 1]) * partion;
}
double TExpressionVariable::Or(const TExpressionVariable& secondOperand) const {
    return !(IsEmpty() && secondOperand.IsEmpty());
}
double TExpressionVariable::And(const TExpressionVariable& secondOperand) const {
    return !(IsEmpty() || secondOperand.IsEmpty());
}
double TExpressionVariable::Cond(const TExpressionVariable& secondOperand, const TExpressionVariable& u) const {
    return !IsEmpty() ? secondOperand.ToDouble() : u.ToDouble();
}
TString TExpressionVariable::StrCond(const TExpressionVariable& secondOperand, const TExpressionVariable& u) const {
    return !IsEmpty() ? secondOperand.ToStr() : u.ToStr();
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
        if (not TryFromString<ui32>(*first, firstValue) || not TryFromString<ui32>(*second, secondValue)){
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
        if (not TryFromString<ui32>(*first, firstValue)){
            return 0.0;
        }
        if (firstValue > 0) {
            return firstG;
        }
        ++first;
    }

    while (second != ss_second.end()) {
        if (not TryFromString<ui32>(*second, secondValue)){
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
        if (not TryFromString<ui32>(*first, firstValue) || \
            not TryFromString<ui32>(*second, secondValue) || \
            firstValue != secondValue){
            return 0.0;
        }
        ++first;
        ++second;
    }

    while (first != ss_first.end()) {
        if (not TryFromString<ui32>(*first, firstValue) || firstValue != 0){
            return 0.0;
        }
        ++first;
    }

    while (second != ss_second.end()) {
        if (not TryFromString<ui32>(*second, secondValue) || secondValue != 0){
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
double TExpressionVariable::E(const TExpressionVariable& secondOperand) const {
    return IsEqual(secondOperand, EPS) ? 1.0 : 0.0;
}
double TExpressionVariable::Ne(const TExpressionVariable& secondOperand) const {
    return IsEqual(secondOperand, EPS) ? 0.0 : 1.0;
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
    if (HasHistogramPointsAndBinsValue) {
        HasDoubleValue = true;
        return double();
    }
    if (HasStrValue) {
        if (HasDoubleValue) {
            return double();
        }

        // try to parse only once
        if (TryFromString<double>(StringValue, DoubleValue)) {
            HasStrValue = false;
            HasDoubleValue = false;
            return DoubleValue;
        }

        HasDoubleValue = true;
        return double();
    }
    return DoubleValue;
}

TString TExpressionVariable::ToStr() const {
    if (HasStrValue) {
        return StringValue;
    }
    if (HasHistogramPointsAndBinsValue) {
        StringValue = ToString(HistogramPointsAndBinsValue);
        return StringValue;
    }
    StringValue = ToString(DoubleValue);
    return StringValue;
}

THistogramPointsAndBins TExpressionVariable::ToHistogramPointsAndBins() const {
    if (HasHistogramPointsAndBinsValue) {
        return HistogramPointsAndBinsValue;
    }
    if (!HasStrValue) {
        return THistogramPointsAndBins();
    }

    if (TryParseFromStringToTHistogramPointsAndBins(HistogramPointsAndBinsValue)) {
        HasStrValue = false;
        HasDoubleValue = true;
        HasHistogramPointsAndBinsValue = true;
        return HistogramPointsAndBinsValue;
    }
    return THistogramPointsAndBins();
}


bool TExpressionVariable::IsEmpty() const {
    return ToDouble() == 0.0;
}

bool TExpressionVariable::TryParse() const {
    if (!HasStrValue && !HasHistogramPointsAndBinsValue)
        return true;

    if (TryFromString<double>(StringValue, DoubleValue)) {
        HasDoubleValue = false;
        HasStrValue = false;
        HasHistogramPointsAndBinsValue = false;
        return true;
    }
    HasDoubleValue = true;
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
    bool compareNumeric = true;

    if (!TryParse()) {
        compareNumeric = false;
    }
    if (!secondOperand.TryParse()) {
        compareNumeric = false;
    }
    if (compareNumeric) {
        return fabs(DoubleValue - secondOperand.DoubleValue) < eps;
    }
    if (HasStrValue && secondOperand.HasStrValue) {
        return StringValue == secondOperand.StringValue;
    }
    if (HasHistogramPointsAndBinsValue && secondOperand.HasHistogramPointsAndBinsValue) {
        return ToString(HistogramPointsAndBinsValue) == ToString(secondOperand.HistogramPointsAndBinsValue);
    }
    return false;
}
