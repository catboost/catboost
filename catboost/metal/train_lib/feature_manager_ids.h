#pragma once

#include <catboost/libs/data/quantized_features_info.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/vector.h>
#include <util/generic/ylimits.h>

namespace NCB {
    struct TMetalOriginalFeatureManagerId {
        ui32 FlatIndex = 0;
        EFeatureType Type = EFeatureType::Float;
        ui32 BorderOffset = 0;
        ui32 BorderCount = 0;
    };

    struct TMetalOriginalFeatureManagerIds {
        TVector<TVector<ui32>> ByFlatIndex;
        TVector<TMetalOriginalFeatureManagerId> ByManagerId;
        ui32 NextId = 0;
    };

    // CUDA binarizations_manager.cpp constructor registers every original
    // Float/Categorical column in external order, including ignored columns.
    // A float occupies one ID per 255-border slice (at least one); original
    // Text/Embedding columns consume no IDs. Estimators register after NextId.
    inline TMetalOriginalFeatureManagerIds MakeMetalOriginalFeatureManagerIds(
            const TFeaturesLayout& layout, const TQuantizedFeaturesInfo& quantizedInfo) {
        TMetalOriginalFeatureManagerIds result;
        const auto& meta = layout.GetExternalFeaturesMetaInfo();
        result.ByFlatIndex.resize(meta.size());
        for (ui32 flat = 0; flat < meta.size(); ++flat) {
            const auto type = meta[flat].Type;
            if (type != EFeatureType::Float && type != EFeatureType::Categorical) continue;
            ui32 borders = 0;
            if (type == EFeatureType::Float) {
                const auto internal = layout.GetInternalFeatureIdx<EFeatureType::Float>(flat);
                if (quantizedInfo.HasBorders(internal)) borders = quantizedInfo.GetBorders(internal).size();
            }
            const ui32 slices = Max<ui32>(1, (ui64(borders) + 254) / 255);
            for (ui32 slice = 0; slice < slices; ++slice) {
                CB_ENSURE(result.NextId < Max<ui32>(), "Metal original feature manager ID count overflows");
                result.ByFlatIndex[flat].push_back(result.NextId++);
                const ui32 offset = slice * 255;
                result.ByManagerId.push_back({flat, type, offset, Min<ui32>(255, borders - offset)});
            }
        }
        return result;
    }
}
