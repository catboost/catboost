#pragma once

#include "feature_manager_ids.h"

#include <catboost/cuda/data/feature.h>
#include <catboost/libs/model/split.h>

#include <algorithm>

namespace NCB {
    // Reconstruct the literal CUDA tensor, then use its actual source hash.
    // Model projections contain original numeric borders/category hashes;
    // CUDA tensors contain manager IDs, slice-local bins and perfect hashes.
    // Selected CTRs contribute their flattened projection, never their own
    // dynamically allocated manager ID (CUDA tree_ctrs.cpp::AddSplit).
    inline NCatboostCuda::TFeatureTensor MakeMetalCudaTreeCtrTensor(
            const TFeatureCombination& projection, const TFeaturesLayout& layout,
            const TQuantizedFeaturesInfo& quantizedInfo,
            const TMetalOriginalFeatureManagerIds& ids) {
        NCatboostCuda::TFeatureTensor tensor;
        for (int cat : projection.CatFeatures) {
            CB_ENSURE(cat >= 0 && ui32(cat) < layout.GetCatFeatureCount(), "Tree CTR base category index is invalid");
            const ui32 flat = layout.GetExternalFeatureIdx(cat, EFeatureType::Categorical);
            CB_ENSURE(ids.ByFlatIndex.at(flat).size() == 1, "Tree CTR base category has no original manager ID");
            tensor.AddCatFeature(ids.ByFlatIndex[flat][0]);
        }
        for (const auto& split : projection.BinFeatures) {
            CB_ENSURE(split.FloatFeature >= 0 && ui32(split.FloatFeature) < layout.GetFloatFeatureCount(),
                "Tree CTR base float index is invalid");
            const TFloatFeatureIdx internal(split.FloatFeature);
            CB_ENSURE(quantizedInfo.HasBorders(internal), "Tree CTR base float has no borders");
            const auto& borders = quantizedInfo.GetBorders(internal);
            const auto found = std::lower_bound(borders.begin(), borders.end(), split.Split);
            CB_ENSURE(found != borders.end() && *found == split.Split, "Tree CTR base predicate is absent from original float borders");
            const ui32 bin = found - borders.begin();
            const ui32 flat = layout.GetExternalFeatureIdx(split.FloatFeature, EFeatureType::Float);
            CB_ENSURE(bin / 255 < ids.ByFlatIndex.at(flat).size(), "Tree CTR base numeric slice has no original manager ID");
            tensor.AddBinarySplit({ids.ByFlatIndex[flat][bin / 255], bin % 255, NCatboostCuda::EBinSplitType::TakeGreater});
        }
        for (const auto& split : projection.OneHotFeatures) {
            CB_ENSURE(split.CatFeatureIdx >= 0 && ui32(split.CatFeatureIdx) < layout.GetCatFeatureCount(),
                "Tree CTR base one-hot index is invalid");
            const ui32 flat = layout.GetExternalFeatureIdx(split.CatFeatureIdx, EFeatureType::Categorical);
            CB_ENSURE(ids.ByFlatIndex.at(flat).size() == 1, "Tree CTR base one-hot has no original manager ID");
            const auto& perfectHash = quantizedInfo.GetCategoricalFeaturesPerfectHash(TCatFeatureIdx(split.CatFeatureIdx));
            const auto value = perfectHash.Find(static_cast<ui32>(split.Value));
            CB_ENSURE(value, "Tree CTR base one-hot value is absent from the original perfect hash");
            tensor.AddBinarySplit({ids.ByFlatIndex[flat][0], value->Value, NCatboostCuda::EBinSplitType::TakeBin});
        }
        return tensor;
    }

    // Tree CTR compressed grids use the CONFIGURED border count, even when
    // the learned grid is constant or has fewer effective borders. Zero-fold
    // configurations are filtered before CUDA policy registration.
    inline ui32 MetalTreeCtrConfiguredPolicy(ui32 borderCount) {
        CB_ENSURE(borderCount > 0 && borderCount <= 255, "Tree CTR scoring policy requires 1..255 configured borders");
        return borderCount <= 1 ? 0 : borderCount <= 15 ? 1 : 2;
    }
}
