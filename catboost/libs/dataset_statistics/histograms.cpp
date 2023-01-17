#include "histograms.h"

#include <util/generic/algorithm.h>
#include <util/generic/size_literals.h>
#include <util/generic/ymath.h>

#include <cmath>

using namespace NCB;

constexpr auto floatInf = std::numeric_limits<float>::infinity();


void TFloatFeatureHistogram::Update(const TFloatFeatureHistogram& histograms) {
    CB_ENSURE_INTERNAL(Borders == histograms.Borders, "Different borders");
    if (!histograms.Histogram.empty()) {
        for (size_t idx = 0; idx < Histogram.size(); ++idx) {
            Histogram[idx] += histograms.Histogram[idx];
        }
    }
    Nans += histograms.Nans;
    MinusInf += histograms.MinusInf;
    PlusInf += histograms.PlusInf;
}

void TFloatFeatureHistogram::CalcUniformHistogram(const TVector<float>& features) {
    Borders.CheckHistogramType(EHistogramType::Uniform);
    Histogram.resize(Borders.MaxBorderCount + 1);

    double length = Borders.MaxValue - Borders.MinValue;
    double step = length / double(Borders.MaxBorderCount);

    for (float value: features) {
        if (ProcessNotNummeric(value)) {
            continue;
        }
        if (value <= Borders.MinValue) {
            ++Histogram[0];
            continue;
        }
        auto index = ui32(double(value - Borders.MinValue) / step);
        if (value > Borders.MinValue + step * index) {
            ++index;
        }
        if (index >= Histogram.size()) {
            index = Histogram.size() - 1;
        }
        ++Histogram[index];
    }
}

bool TFloatFeatureHistogram::ProcessNotNummeric(float value) {
    if (std::isnan(value)) {
        ++Nans;
        return true;
    }
    if (std::isinf(value)) {
        if (value == floatInf) {
            ++PlusInf;
        } else {
            ++MinusInf;
        }
        return true;
    }
    return false;
}

void TFloatFeatureHistogram::CalcHistogramWithBorders(
    const TVector<float>& featureColumnPtr
) {
    Borders.CheckHistogramType(EHistogramType::Borders);
    TVector<float> featureColumnCopy(featureColumnPtr);
    CalcHistogramWithBorders(&featureColumnCopy);
}

void TFloatFeatureHistogram::CalcHistogramWithBorders(
    TVector<float>* featureColumnPtr
) {
    Borders.CheckHistogramType(EHistogramType::Borders);
    TVector<float>& features = *featureColumnPtr;
    Histogram.resize(Borders.Size());
    std::sort(features.begin(), features.end());
    ui32 borderIndex = 0;
    for (auto value: features) {
        if (ProcessNotNummeric(value)) {
            continue;
        }
        while (borderIndex + 1 < Histogram.size() && value > Borders.Borders[borderIndex]) {
            ++borderIndex;
        }
        Histogram[borderIndex]++;
    }
}

NJson::TJsonValue TBorders::ToJson() const {
    NJson::TJsonValue result;

    InsertEnumType("HistogramType", HistogramType, &result);
    switch (HistogramType) {
        case EHistogramType::Undefined:
            break;
        case EHistogramType::Uniform:
            result.InsertValue("MaxBorderCount", MaxBorderCount);
            result.InsertValue("MinValue", MinValue);
            result.InsertValue("MaxValue", MaxValue);
            break;
        case EHistogramType::Borders:
            result.InsertValue("Borders", VectorToJson(Borders));
            break;
        default:
            Y_ASSERT(false);
    }
    return result;
}

NJson::TJsonValue TFloatFeatureHistogram::ToJson() const {
    NJson::TJsonValue result;

    result.InsertValue("Nans", Nans);
    result.InsertValue("MinusInf", MinusInf);
    result.InsertValue("PlusInf", PlusInf);
    result.InsertValue("Histogram", VectorToJson(Histogram));
    result.InsertValue("Borders", Borders.ToJson());
    return result;
}

NJson::TJsonValue THistograms::ToJson() const {
    NJson::TJsonValue result;
    TVector<NJson::TJsonValue> histogram;
    for (const auto& item : FloatFeatureHistogram) {
        histogram.emplace_back(item.ToJson());
    }
    result.InsertValue("FloatFeatureHistogram", VectorToJson(histogram));
    return result;
}

void THistograms::Update(const THistograms& histograms) {
    Y_ASSERT(FloatFeatureHistogram.size() == histograms.FloatFeatureHistogram.size());
    for (size_t idx = 0; idx < FloatFeatureHistogram.size(); ++idx) {
        FloatFeatureHistogram[idx].Update(histograms.FloatFeatureHistogram[idx]);
    }
}

void THistograms::AddFloatFeatureUniformHistogram(
    ui32 featureId,
    TVector<float>* features
) {
    CB_ENSURE_INTERNAL(
        featureId < FloatFeatureHistogram.size(),
        "FeaturedId " << featureId << " is bigger then FloatFeatureHistogram size " << FloatFeatureHistogram.size()
    );
    FloatFeatureHistogram[featureId].CalcUniformHistogram(*features);
}

bool TBorders::operator==(const TBorders& rhs) {
    if (HistogramType != rhs.HistogramType) {
        return false;
    }
    switch (HistogramType) {
        case EHistogramType::Undefined:
            return true;
        case EHistogramType::Uniform:
            return MaxBorderCount == rhs.MaxBorderCount && MinValue == rhs.MinValue && MaxValue == rhs.MaxValue;
        case EHistogramType::Borders:
            return EqualBorders(rhs.Borders);
        default:
            Y_ASSERT(false);
    }
    return false;
}

ui32 TBorders::Size() const {
    switch (HistogramType) {
        case EHistogramType::Undefined:
            return 0;
        case EHistogramType::Uniform:
            return MaxBorderCount;
        case EHistogramType::Borders:
            return Borders.size();
        default:
            Y_ASSERT(false);
    }
    return 0;
}

static TVector<float> GetUniformBorders(
    ui32 maxBorderCount,
    float minValue,
    float maxValue
) {
    if (fabs(minValue - maxValue) < 1e-9) {
        return {minValue};
    }
    CB_ENSURE(minValue < maxValue, "Min value > Max value: " << minValue << ">" << maxValue);
    TVector<float> borders(maxBorderCount + 1);
    borders[0] = minValue;
    borders[maxBorderCount] = maxValue;
    double step = (maxValue - minValue) / double(maxBorderCount);
    for (ui32 idx = 1; idx < maxBorderCount; ++idx) {
        borders[idx] = minValue + float(step * double(idx));
    }
    return borders;
}

TVector<float> TBorders::GetBorders() const {
    switch (HistogramType) {
        case EHistogramType::Undefined:
            CB_ENSURE(false, "No borders");
        case EHistogramType::Uniform:
            return GetUniformBorders(MaxBorderCount, MinValue, MaxValue);
        case EHistogramType::Borders:
            return Borders;
        default:
            Y_ASSERT(false);
    }
    return {};
}
