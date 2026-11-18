#pragma once

#include "feature_manager_ids.h"

#include <util/generic/array_ref.h>

namespace NCB {
    // The GPU option contains CUDA feature-manager IDs, not dense Metal
    // columns or float-only indices. Resolve before sending the native ABI.
    // Symmetric scalar/query CUDA trainers ignore the option; the symmetric
    // multiclass-family greedy template rejects it. The caller applies that
    // policy gate before calling this helper for non-symmetric trees only.
    inline TVector<ui32> ResolveMetalFixedBinarySplits(
            TConstArrayRef<ui32> requested,
            const TFeaturesLayout& layout,
            const TQuantizedFeaturesInfo& quantizedInfo,
            TConstArrayRef<ui32> denseFloatFlatIndices) {
        const auto ids = MakeMetalOriginalFeatureManagerIds(layout, quantizedInfo);
        TVector<ui32> denseByFlat(ids.ByFlatIndex.size(), Max<ui32>());
        for (ui32 dense = 0; dense < denseFloatFlatIndices.size(); ++dense) {
            const ui32 flat = denseFloatFlatIndices[dense];
            CB_ENSURE(flat < denseByFlat.size() && denseByFlat[flat] == Max<ui32>(),
                "Metal dense numeric feature map is invalid");
            denseByFlat[flat] = dense;
        }
        TVector<ui32> result;
        result.reserve(requested.size());
        for (ui32 id : requested) {
            CB_ENSURE(id < ids.ByManagerId.size(), "Fixed split feature " << id << " is not a float feature");
            const auto& feature = ids.ByManagerId[id];
            const auto& meta = layout.GetExternalFeatureMetaInfo(feature.FlatIndex);
            CB_ENSURE(feature.Type == EFeatureType::Float && meta.IsAvailable && !meta.IsIgnored,
                "Fixed splits are supported only for available float features. Feature " << id << " is not a float feature");
            CB_ENSURE(feature.BorderCount == 1,
                "Fixed splits are supported only for binary features. Feature " << id << " has " << feature.BorderCount << " borders");
            // Prepared Metal numeric columns currently support at most 255
            // borders; do not silently route the final slice of a wider float
            // through bin zero of the unsliced column.
            CB_ENSURE(feature.BorderOffset == 0 && ids.ByFlatIndex[feature.FlatIndex].size() == 1,
                "Metal fixed splits require an unsliced binary float feature");
            CB_ENSURE(denseByFlat[feature.FlatIndex] != Max<ui32>(),
                "Fixed split feature " << id << " has no prepared numeric column");
            result.push_back(denseByFlat[feature.FlatIndex]);
        }
        return result;
    }
}
