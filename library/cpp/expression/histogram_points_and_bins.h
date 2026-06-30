#pragma once

#include <numeric>

#include <util/generic/vector.h>
#include <util/string/builder.h>
#include <util/string/cast.h>


class THistogramPointsAndBins {
    private:
        TVector<double> Points;
        TVector<double> Bins;
    public:
        THistogramPointsAndBins();
        THistogramPointsAndBins(const TVector<double>& points, const TVector<double>& bins);

        const TVector<double>& GetPoints() const;
        const TVector<double>& GetBins() const;
        void SetPointsAndBins(const TVector<double>& points, const TVector<double>& bins);
        const std::pair<size_t, double> FindBinAndPartion(const double& percentile) const;
        bool IsEqual(const THistogramPointsAndBins& secondOperand, const double eps) const;
        bool IsValidBins() const;
        bool IsValidPoints() const;
        bool IsValidPercentile(const double& percentile) const;
        bool IsValidData(const double& percentile) const;
};
