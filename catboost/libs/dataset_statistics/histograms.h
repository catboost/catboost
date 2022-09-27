#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/json_helpers.h>

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/vector.h>
#include <util/ysaveload.h>

namespace  NCB {
enum class EHistogramType {
    Uniform,   // (-inf, MinValue], (MinValue, MinValue + step], ... , (MaxValue - step, MaxValue]
    Borders,
    Undefined
};

struct TBorders {
public:
    TBorders()
        : HistogramType(EHistogramType::Undefined)
    {}

    TVector<float> GetBorders() const;

    bool operator==(const TBorders& rhs);

    TBorders(const TVector<float>& borders)
        : HistogramType(EHistogramType::Borders)
        , Borders(borders)
    {}

    TBorders(ui32 maxBorderCount, float minValue, float maxValue)
        : HistogramType(EHistogramType::Uniform)
        , MaxBorderCount(maxBorderCount)
        , MinValue(minValue)
        , MaxValue(maxValue)
    {}

    ui32 Size() const;

    Y_SAVELOAD_DEFINE(
        HistogramType,
        Borders,
        MaxBorderCount,
        MinValue,
        MaxValue
    );

    SAVELOAD(
        HistogramType,
        Borders,
        MaxBorderCount,
        MinValue,
        MaxValue
    );

    NJson::TJsonValue ToJson() const;

    void CheckHistogramType(EHistogramType type) const {
        CB_ENSURE_INTERNAL(
            HistogramType == type,
            "Inconsistent HistogramType " << HistogramType << " != " << type;
        );
    }

private:
    bool EqualBorders(const TVector<float>& borders) {
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
    TVector<float> Borders;
    ui32 MaxBorderCount;
    float MinValue;
    float MaxValue;
};

struct TFloatFeatureHistogram {
public:
    TFloatFeatureHistogram()
        : Nans(0), MinusInf(0), PlusInf(0)
    {}

    TFloatFeatureHistogram(const TBorders& borders)
        : Borders(borders), Nans(0), MinusInf(0), PlusInf(0)
    {}

    void Update(const TFloatFeatureHistogram &histograms);

    void CalcUniformHistogram(const TVector<float>& features);

    void CalcHistogramWithBorders(const TVector<float>& featureColumn);

    // featuresColumn will be shuffled after call
    void CalcHistogramWithBorders(TVector<float>* featureColumnPtr);

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

    NJson::TJsonValue ToJson() const;

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

    void Update(const THistograms& histograms);

    void AddFloatFeatureUniformHistogram(
        ui32 featureId,
        TVector<float>* features
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
}