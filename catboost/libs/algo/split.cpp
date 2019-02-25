#include "split.h"

#include "learn_context.h"
#include "online_ctr.h"

#include <catboost/libs/data_new/packed_binary_features.h>

#include <util/system/yassert.h>

#include <climits>


using namespace NCB;


const size_t TSplitCandidate::FloatFeatureBaseHash = 12321;
const size_t TSplitCandidate::CtrBaseHash = 89321;
const size_t TSplitCandidate::OneHotFeatureBaseHash = 517931;

TModelSplit TSplit::GetModelSplit(
    const TLearnContext& ctx,
    const TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap
) const {
    TModelSplit split;
    split.Type = Type;
    if (Type == ESplitType::FloatFeature) {
        split.FloatFeature.FloatFeature = FeatureIdx;
        split.FloatFeature.Split = ctx.LearnProgress.FloatFeatures[FeatureIdx].Borders[BinBorder];
    } else if (Type == ESplitType::OneHotFeature) {
        split.OneHotFeature.CatFeatureIdx = FeatureIdx;
        split.OneHotFeature.Value = perfectHashedToHashedCatValuesMap[FeatureIdx][BinBorder];
    } else {
        Y_ASSERT(Type == ESplitType::OnlineCtr);
        auto& ctrBase = split.OnlineCtr.Ctr.Base;
        auto& featureCombination = ctrBase.Projection;
        featureCombination.CatFeatures = Ctr.Projection.CatFeatures;
        for (auto binFeature : Ctr.Projection.BinFeatures) {
            auto& ref = featureCombination.BinFeatures.emplace_back();
            ref.FloatFeature = binFeature.FloatFeature;
            ref.Split = ctx.LearnProgress.FloatFeatures[binFeature.FloatFeature].Borders[binFeature.SplitIdx];
        }
        for (auto oheFeature : Ctr.Projection.OneHotFeatures) {
            auto& ref = featureCombination.OneHotFeatures.emplace_back();
            ref.CatFeatureIdx = oheFeature.CatFeatureIdx;
            ref.Value = perfectHashedToHashedCatValuesMap[oheFeature.CatFeatureIdx][oheFeature.Value];
        }
        auto& ctrHelper = ctx.CtrsHelper;
        const auto ctrIdx = Ctr.CtrIdx;
        const auto& ctrInfo = ctrHelper.GetCtrInfo(Ctr.Projection)[ctrIdx];
        const TVector<float>& priors = ctrInfo.Priors;

        TVector<float> shift;
        TVector<float> norm;
        CalcNormalization(priors, &shift, &norm);
        ctrBase.CtrType = ctrInfo.Type;
        ctrBase.TargetBorderClassifierIdx = ctrInfo.TargetClassifierIdx;
        split.OnlineCtr.Ctr.TargetBorderIdx = Ctr.TargetBorderIdx;
        split.OnlineCtr.Ctr.PriorNum = priors[Ctr.PriorIdx];
        split.OnlineCtr.Ctr.PriorDenom = 1.0f;
        split.OnlineCtr.Ctr.Shift = shift[Ctr.PriorIdx];
        split.OnlineCtr.Ctr.Scale = ctrInfo.BorderCount / norm[Ctr.PriorIdx];
        split.OnlineCtr.Border = EmulateUi8Rounding(BinBorder);
    }
    return split;
}


int GetBucketCount(
    const TSplitEnsemble& splitEnsemble,
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
    size_t packedBinaryFeaturesCount
) {
    if (splitEnsemble.IsBinarySplitsPack) {
        // TBinarySplitsPack
        size_t packIdx = splitEnsemble.BinarySplitsPack.PackIdx;
        size_t startIdx = packIdx * sizeof(TBinaryFeaturesPack) * CHAR_BIT;
        Y_ASSERT(packedBinaryFeaturesCount > startIdx);
        size_t featuresInPackCount = Min(
            sizeof(TBinaryFeaturesPack) * CHAR_BIT,
            packedBinaryFeaturesCount - startIdx
        );
        return int(1 << featuresInPackCount);
    }

    // TSplitCandidate
    const auto& splitCandidate = splitEnsemble.SplitCandidate;
    if (splitCandidate.Type == ESplitType::OnlineCtr) {
        return splitCandidate.Ctr.BorderCount + 1;
    } else if (splitCandidate.Type == ESplitType::FloatFeature) {
        return int(
            quantizedFeaturesInfo.GetBorders(TFloatFeatureIdx(splitCandidate.FeatureIdx)).size()
        ) + 1;
    } else {
        Y_ASSERT(splitCandidate.Type == ESplitType::OneHotFeature);
        return int(
            quantizedFeaturesInfo.GetUniqueValuesCounts(TCatFeatureIdx(splitCandidate.FeatureIdx))
                .OnLearnOnly
        );
    }
}
