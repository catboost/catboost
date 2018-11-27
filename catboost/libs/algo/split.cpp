#include "split.h"
#include "learn_context.h"

const size_t TSplitCandidate::FloatFeatureBaseHash = 12321;
const size_t TSplitCandidate::CtrBaseHash = 89321;
const size_t TSplitCandidate::OneHotFeatureBaseHash = 517931;

TModelSplit TSplit::GetModelSplit(const TLearnProgress& learnProgress, const TCtrHelper& ctrHelper, const TDataset& learnData) const   {
    TModelSplit split;
    split.Type = Type;
    if (Type == ESplitType::FloatFeature) {
        split.FloatFeature.FloatFeature = FeatureIdx;
        split.FloatFeature.Split = learnProgress.FloatFeatures[FeatureIdx].Borders[BinBorder];
    } else if (Type == ESplitType::OneHotFeature) {
        split.OneHotFeature.CatFeatureIdx = FeatureIdx;
        split.OneHotFeature.Value = learnData.AllFeatures.OneHotValues[FeatureIdx][BinBorder];
    } else {
        Y_ASSERT(Type == ESplitType::OnlineCtr);
        auto& ctrBase = split.OnlineCtr.Ctr.Base;
        auto& featureCombination = ctrBase.Projection;
        featureCombination.CatFeatures = Ctr.Projection.CatFeatures;
        for (auto binFeature : Ctr.Projection.BinFeatures) {
            auto& ref = featureCombination.BinFeatures.emplace_back();
            ref.FloatFeature = binFeature.FloatFeature;
            ref.Split = learnProgress.FloatFeatures[binFeature.FloatFeature].Borders[binFeature.SplitIdx];
        }
        for (auto oheFeature : Ctr.Projection.OneHotFeatures) {
            auto& ref = featureCombination.OneHotFeatures.emplace_back();
            ref.CatFeatureIdx = oheFeature.CatFeatureIdx;
            ref.Value = learnData.AllFeatures.OneHotValues[oheFeature.CatFeatureIdx][oheFeature.Value];
        }
        const auto ctrIdx = Ctr.CtrIdx;
        const auto& ctrInfo =  ctrHelper.GetCtrInfo(Ctr.Projection)[ctrIdx];
        const TVector<float>& priors =  ctrInfo.Priors;


        TVector<float> shift;
        TVector<float> norm;
        CalcNormalization(priors, &shift, &norm);
        ctrBase.CtrType = ctrInfo.Type;
        ctrBase.TargetBorderClassifierIdx  = ctrInfo.TargetClassifierIdx;
        split.OnlineCtr.Ctr.TargetBorderIdx  = Ctr.TargetBorderIdx;
        split.OnlineCtr.Ctr.PriorNum = priors[Ctr.PriorIdx];
        split.OnlineCtr.Ctr.PriorDenom = 1.0f;
        split.OnlineCtr.Ctr.Shift = shift[Ctr.PriorIdx];
        split.OnlineCtr.Ctr.Scale = ctrInfo.BorderCount / norm[Ctr.PriorIdx];
        split.OnlineCtr.Border = EmulateUi8Rounding(BinBorder);
    }
    return split;
}


int GetSplitCount(
    const TVector<int>& splitsCount,
    const TVector<TVector<int>>& oneHotValues,
    const TSplitCandidate& split
) {
    if (split.Type == ESplitType::OnlineCtr) {
        return split.Ctr.BorderCount;
    } else if (split.Type == ESplitType::FloatFeature) {
        return splitsCount[split.FeatureIdx];
    } else {
        Y_ASSERT(split.Type == ESplitType::OneHotFeature);
        return oneHotValues[split.FeatureIdx].ysize();
    }
}
