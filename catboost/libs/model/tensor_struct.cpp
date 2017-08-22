#include "tensor_struct.h"

yvector<TBinFeature> GetBinFeatures(const TTensorStructure3& tree) {
    yvector<TBinFeature> result;
    for (const auto& split : tree.SelectedSplits) {
        if (split.Type == ESplitType::FloatFeature) {
            result.push_back(split.BinFeature);
        }
    }
    return result;
}

yvector<TOneHotFeature> GetOneHotFeatures(const TTensorStructure3& tree) {
    yvector<TOneHotFeature> result;
    for (const auto& split : tree.SelectedSplits) {
        if (split.Type == ESplitType::OneHotFeature) {
            result.push_back(split.OneHotFeature);
        }
    }
    return result;
}

yvector<TModelCtrSplit> GetCtrSplits(const TTensorStructure3& tree) {
    yvector<TModelCtrSplit> result;
    for (const auto& split : tree.SelectedSplits) {
        if (split.Type == ESplitType::OnlineCtr) {
            result.push_back(split.OnlineCtr);
        }
    }
    return result;
}
