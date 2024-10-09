#include "histogram_points_and_bins.h"


THistogramPointsAndBins::THistogramPointsAndBins()
    : Points()
    , Bins()
    {
    }

THistogramPointsAndBins::THistogramPointsAndBins(const TVector<double>& points, const TVector<double>& bins) {
    if (points.size() == (bins.size() - 1)) {
        Points = points;
        Bins = bins;
    }
}

bool THistogramPointsAndBins::operator==(const THistogramPointsAndBins& secondOperand) const {
    return Points == secondOperand.GetPoints() && Bins == secondOperand.GetBins();
}

const TVector<double>& THistogramPointsAndBins::GetPoints() const {
    return Points;
}

const TVector<double>& THistogramPointsAndBins::GetBins() const {
    return Bins;
}

void THistogramPointsAndBins::SetPointsAndBins(const TVector<double>& points, const TVector<double>& bins) {
    if (points.size() == (bins.size() - 1)) {
        Points = points;
        Bins = bins;
    }
}

const std::pair<size_t, double> THistogramPointsAndBins::FindBinAndPartion(const double& percentile) const {
    double targetSum = std::accumulate(Bins.begin(), Bins.end(), 0.0) * percentile / 100;
    double currentSum = 0.0;

    for (size_t i = 0; i < Bins.size(); ++i) {
        currentSum += Bins[i];
        if (currentSum >= targetSum) {
            return {i, 1.0 - (currentSum - targetSum) / Bins[i]};
        }
    }
    return {Bins.size() - 1, 1.0};
}

bool THistogramPointsAndBins::IsBinsFilledWithZeros() const {
    for (const auto& bin: Bins) {
        if (bin != 0.0) {
            return false;
        }
    }
    return true;
}

bool THistogramPointsAndBins::IsEmptyData() const {
    return Points.size() == 0 || Bins.size() == 0;
}

bool THistogramPointsAndBins::IsInvalidPercentile(const double& percentile) const {
    return percentile <= 0.0 || 100.0 <= percentile;
}

bool THistogramPointsAndBins::IsInvalidData(const double& percentile) const {
    return IsBinsFilledWithZeros() || IsEmptyData() || IsInvalidPercentile(percentile);
}

template <>
void Out<THistogramPointsAndBins>(IOutputStream& o, const THistogramPointsAndBins& pointsAndBins) {
    for (const auto& point: pointsAndBins.GetPoints()) {
        o << std::to_string(point) << ",";
    }
    o << ";";
    for (const auto& bin: pointsAndBins.GetBins()) {
        o << std::to_string(bin) << ",";
    }
}
