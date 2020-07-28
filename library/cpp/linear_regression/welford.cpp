#include "welford.h"

#include <util/generic/ymath.h>

void TMeanCalculator::Multiply(const double value) {
    SumWeights *= value;
}

void TMeanCalculator::Add(const double value, const double weight /*= 1.*/) {
    SumWeights += weight;
    if (SumWeights.Get()) {
        Mean += weight * (value - Mean) / SumWeights.Get();
    }
}

void TMeanCalculator::Remove(const double value, const double weight /*= 1.*/) {
    SumWeights -= weight;
    if (SumWeights.Get()) {
        Mean -= weight * (value - Mean) / SumWeights.Get();
    }
}

double TMeanCalculator::GetMean() const {
    return Mean;
}

double TMeanCalculator::GetSumWeights() const {
    return SumWeights.Get();
}

void TMeanCalculator::Reset() {
    *this = TMeanCalculator();
}

void TCovariationCalculator::Add(const double firstValue, const double secondValue, const double weight /*= 1.*/) {
    SumWeights += weight;
    if (SumWeights.Get()) {
        FirstValueMean += weight * (firstValue - FirstValueMean) / SumWeights.Get();
        Covariation += weight * (firstValue - FirstValueMean) * (secondValue - SecondValueMean);
        SecondValueMean += weight * (secondValue - SecondValueMean) / SumWeights.Get();
    }
}

void TCovariationCalculator::Remove(const double firstValue, const double secondValue, const double weight /*= 1.*/) {
    SumWeights -= weight;
    if (SumWeights.Get()) {
        FirstValueMean -= weight * (firstValue - FirstValueMean) / SumWeights.Get();
        Covariation -= weight * (firstValue - FirstValueMean) * (secondValue - SecondValueMean);
        SecondValueMean -= weight * (secondValue - SecondValueMean) / SumWeights.Get();
    }
}

double TCovariationCalculator::GetFirstValueMean() const {
    return FirstValueMean;
}

double TCovariationCalculator::GetSecondValueMean() const {
    return SecondValueMean;
}

double TCovariationCalculator::GetCovariation() const {
    return Covariation;
}

double TCovariationCalculator::GetSumWeights() const {
    return SumWeights.Get();
}

void TCovariationCalculator::Reset() {
    *this = TCovariationCalculator();
}

void TDeviationCalculator::Add(const double value, const double weight /*= 1.*/) {
    const double lastMean = MeanCalculator.GetMean();
    MeanCalculator.Add(value, weight);
    Deviation += weight * (value - lastMean) * (value - MeanCalculator.GetMean());
}

void TDeviationCalculator::Remove(const double value, const double weight /*= 1.*/) {
    const double lastMean = MeanCalculator.GetMean();
    MeanCalculator.Remove(value, weight);
    Deviation -= weight * (value - lastMean) * (value - MeanCalculator.GetMean());
}

double TDeviationCalculator::GetMean() const {
    return MeanCalculator.GetMean();
}

double TDeviationCalculator::GetDeviation() const {
    return Deviation;
}

double TDeviationCalculator::GetStdDev() const {
    const double sumWeights = GetSumWeights();
    if (!sumWeights) {
        return 0.;
    }
    return sqrt(GetDeviation() / sumWeights);
}

double TDeviationCalculator::GetSumWeights() const {
    return MeanCalculator.GetSumWeights();
}

void TDeviationCalculator::Reset() {
    *this = TDeviationCalculator();
}
