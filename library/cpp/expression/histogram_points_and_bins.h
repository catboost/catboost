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
        THistogramPointsAndBins(TVector<double>& points, TVector<double>& bins);

        const TVector<double>& GetPoints() const;
        const TVector<double>& GetBins() const;
        void SetPointsAndBins(TVector<double>& points, TVector<double>& bins);
        const std::pair<size_t, double> FindBinAndPartion(const double& percentile) const;
        bool IsBinsFilledWithZeros() const;
        bool IsEmptyData() const;
        bool IsInvalidPercentile(const double& percentile) const;
        bool IsInvalidData(const double& percentile) const;
};
