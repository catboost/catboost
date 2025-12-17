#include "gpu_binarization_helpers.h"
#include "kernels.h"
#include <catboost/private/libs/quantization/utils.h>

#include <util/generic/algorithm.h>

#include <cmath>

inline TVector<float> ComputeBorders(const NCatboostOptions::TBinarizationOptions& binarizationDescription,
                                     const TSingleBuffer<const float>& floatFeature, ui32 stream) {
    const ui32 borderCount = binarizationDescription.BorderCount.Get();
    CB_ENSURE(borderCount > 0, "Border count should be > 0");

    auto bordersGpu = TSingleBuffer<float>::Create(floatFeature.GetMapping().RepeatOnAllDevices(borderCount + 1));
    ComputeBordersOnDevice(floatFeature, binarizationDescription, bordersGpu, stream);

    TVector<float> bordersWithHeader;
    bordersWithHeader.yresize(borderCount + 1);
    bordersGpu.Read(bordersWithHeader, stream);

    TVector<float> borders;
    borders.assign(bordersWithHeader.begin() + 1, bordersWithHeader.end());
    EraseIf(borders, [] (float v) { return !std::isfinite(v); });
    SortUnique(borders);

    if (borders.empty()) {
        borders.push_back(0.5f);
    }
    return borders;
}


TVector<float> NCatboostCuda::TGpuBordersBuilder::GetOrComputeBorders(
    ui32 featureId,
    const NCatboostOptions::TBinarizationOptions& binarizationDescription,
    TSingleBuffer<const float> floatFeature,
    ui32 stream) {
        TVector<float> borders;
        bool hasBorders = false;
        with_lock (Lock) {
            if (FeaturesManager.HasBorders(featureId)) {
                borders = FeaturesManager.GetBorders(featureId);
                hasBorders = true;
            }
        }
        if (!hasBorders) {
            borders = ComputeBorders(binarizationDescription, floatFeature, stream);
            with_lock (Lock) {
                if (FeaturesManager.HasBorders(featureId)) {
                    borders = FeaturesManager.GetBorders(featureId);
                } else {
                    FeaturesManager.SetBorders(featureId, borders);
                }
            }
        }
        return borders;
}


TVector<float> NCatboostCuda::TGpuBordersBuilder::GetOrComputeBorders(
    ui32 featureId,
    const NCatboostOptions::TBinarizationOptions& binarizationDescription,
    TConstArrayRef<float> floatFeature,
    ui32 stream) {
    TVector<float> borders;
    bool hasBorders = false;
    with_lock (Lock) {
        if (FeaturesManager.HasBorders(featureId)) {
            borders = FeaturesManager.GetBorders(featureId);
            hasBorders = true;
        }
    }
    if (hasBorders) {
        return borders;
    } else {
        TSingleBuffer<float> floatFeatureGpu = TSingleBuffer<float>::Create(NCudaLib::TSingleMapping(0, floatFeature.size()));
        floatFeatureGpu.Write(floatFeature, stream);
        return GetOrComputeBorders(featureId, binarizationDescription, floatFeatureGpu.AsConstBuf(), stream);
    }
}
