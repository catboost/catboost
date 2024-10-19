#include "histogram_points_and_bins.h"


THistogramPointsAndBins::THistogramPointsAndBins()
    : Points()
    , Bins()
    {
    }

THistogramPointsAndBins::THistogramPointsAndBins(const TVector<double>& points, const TVector<double>& bins) {
    SetPointsAndBins(points, bins);
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

bool THistogramPointsAndBins::IsEqual(const THistogramPointsAndBins& secondOperand, const double eps) const {
    if (Points.size() != secondOperand.GetPoints().size() || Bins.size() != secondOperand.GetBins().size()) {
        return false;
    }
    for (size_t i = 0; i < Points.size(); i ++) {
        if (fabs(Points[i] - secondOperand.GetPoints()[i]) >= eps) {
            return false;
        }
    }
    for (size_t i = 0; i < Bins.size(); i ++) {
        if (fabs(Bins[i] - secondOperand.GetBins()[i]) >= eps) {
            return false;
        }
    }
    return true;
}

bool THistogramPointsAndBins::IsValidBins() const {
    bool hasPositiveBinValue = false;
    for (const auto& bin: Bins) {
        if (bin < 0) {
            return false;
        } else if (!hasPositiveBinValue && bin > 0) {
            hasPositiveBinValue = true;
        }
    }
    return hasPositiveBinValue;
}

bool THistogramPointsAndBins::IsValidPoints() const {
    for (size_t i = 1; i < Points.size(); i ++) {
        if (Points[i - 1] >= Points[i]) {
            return false;
        }
    }
    return Points.size() != 0;
}

bool THistogramPointsAndBins::IsValidPercentile(const double& percentile) const {
    return 0 < percentile && percentile < 100;
}

bool THistogramPointsAndBins::IsValidData(const double& percentile) const {
    return IsValidBins() && IsValidPoints() && IsValidPercentile(percentile);
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
