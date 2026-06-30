#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/json_helpers.h>
#include <catboost/private/libs/options/enums.h>

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/map.h>
#include <util/ysaveload.h>

#include <cmath>
#include <limits>


namespace  NCB {

struct TFeatureStatistics;


enum class EHistogramType {
    Uniform,   // (-inf, MinValue], (MinValue, MinValue + step], ... , (MaxValue - step, MaxValue]
    Exact,
    Borders
};

constexpr ui32 MAX_EXACT_HIST_SIZE = 1 << 8;

struct TBorders {
public:
    TBorders()
        : HistogramType(EHistogramType::Exact)
        , OutOfDomainValuesCount(0)
    {}

    TBorders(const TVector<float>& borders)
        : HistogramType(EHistogramType::Borders)
        , Borders(borders)
        , OutOfDomainValuesCount(0)
    {}

    TBorders(ui32 maxBorderCount, float minValue, float maxValue)
        : HistogramType(EHistogramType::Exact)
        , MaxBorderCount(maxBorderCount)
        , MinValue(minValue)
        , MaxValue(maxValue)
        , OutOfDomainValuesCount(0)
    {}

    bool operator==(const TBorders& rhs) const;

    TVector<float> GetBorders() const;
    TVector<float> GetBins() const;
    TVector<ui64> GetExactHistogram() const;

    ui32 Size() const;

    Y_SAVELOAD_DEFINE(
        HistogramType,
        Borders,
        MaxBorderCount,
        MinValue,
        MaxValue,
        BitHistogram,
        OutOfDomainValuesCount
    );

    SAVELOAD(
        HistogramType,
        Borders,
        MaxBorderCount,
        MinValue,
        MaxValue,
        BitHistogram,
        OutOfDomainValuesCount
    );

    NJson::TJsonValue ToJson() const;

    void CheckHistogramType(EHistogramType type) const {
        CB_ENSURE_INTERNAL(
            HistogramType == type,
            "Inconsistent HistogramType " << HistogramType << " != " << type;
        );
    }

private:
    bool EqualBorders(const TVector<float>& borders) const {
        CB_ENSURE(HistogramType == EHistogramType::Borders, "Inconsistent type");
        if (Borders.size() != borders.size()) {
            return false;
        }
        for (size_t idx = 0; idx < Borders.size(); ++idx) {
            if (fabs(Borders[idx] - borders[idx]) > 1e-6) {
                return false;
            }
        }
        return true;
    }

    bool EqualUniformOptions(ui32 maxBorderCount, float minValue, float maxValue) {
        CB_ENSURE(HistogramType == EHistogramType::Uniform, "Inconsistent type");
        return maxBorderCount == MaxBorderCount && minValue == MinValue && maxValue == MaxValue;
    }

public:
    EHistogramType HistogramType;
    TVector<float> Borders;         // valid only if HistogramType == EHistogramType::Borders
    ui32 MaxBorderCount = 0;
    float MinValue = std::numeric_limits<float>::quiet_NaN();
    float MaxValue = std::numeric_limits<float>::quiet_NaN();
    TMap<float, ui64> BitHistogram; // valid only if HistogramType == EHistogramType::Exact
    ui64 OutOfDomainValuesCount = 0;
};

struct TFloatFeatureHistogram {
public:
    TFloatFeatureHistogram()
        : Nans(0)
        , MinusInf(0)
        , PlusInf(0)
    {}

    TFloatFeatureHistogram(const TBorders& borders)
        : Borders(borders)
        , Nans(0)
        , MinusInf(0)
        , PlusInf(0)
    {}

    bool operator==(const TFloatFeatureHistogram& a) const {
        return std::tie(Histogram, Borders, Nans, MinusInf, PlusInf) ==
               std::tie(a.Histogram, a.Borders, a.Nans, a.MinusInf, a.PlusInf);
    }

    void Update(TFloatFeatureHistogram &histograms);

    void CalcUniformHistogram(TConstArrayRef<float> features, const TVector<ui64>& count={});

    void CalcHistogramWithBorders(const TVector<float>& featureColumn);

    // featuresColumn will be shuffled after call
    void CalcHistogramWithBorders(TVector<float>* featureColumnPtr);

    TVector<ui64> GetHistogram() const;

    void ConvertBitToUniformIfNeeded();

    void ConvertBitToUniform();

    NJson::TJsonValue ToJson() const;

    Y_SAVELOAD_DEFINE(
        Histogram,
        Borders,
        Nans,
        MinusInf,
        PlusInf
    );

    SAVELOAD(
        Histogram,
        Borders,
        Nans,
        MinusInf,
        PlusInf
    );

private:
    bool ProcessNotNummeric(float f);

public:
    TVector<ui64> Histogram;
    TBorders Borders;

    ui64 Nans;
    ui64 MinusInf;
    ui64 PlusInf;
};

struct THistograms {
public:
    THistograms() = default;

    THistograms(const TVector<TBorders>& floatFeatureBorders) {
        FloatFeatureHistogram.reserve(floatFeatureBorders.size());
        for (const auto& borders: floatFeatureBorders) {
            FloatFeatureHistogram.emplace_back(TFloatFeatureHistogram(borders));
        }
    }

    NJson::TJsonValue ToJson() const;

    void Update(THistograms& histograms);

    void AddFloatFeatureUniformHistogram(
        ui32 featureId,
        TConstArrayRef<float> features
    );

    Y_SAVELOAD_DEFINE(
        FloatFeatureHistogram
    );

    SAVELOAD(
        FloatFeatureHistogram
    );

    ui64 GetObjectCount() const {
        if (FloatFeatureHistogram.empty()) {
            return 0;
        }
        ui64 result = 0;
        for (auto bucket : FloatFeatureHistogram[0].Histogram) {
            result += bucket;
        }
        return result;
    }

public:
    TVector<TFloatFeatureHistogram> FloatFeatureHistogram;
};

THistograms InitHistograms(const TVector<size_t>& borderCounts, const TFeatureStatistics& featuresStatistics);

}
