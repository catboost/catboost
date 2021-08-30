#include "grid.h"

namespace NMonoForest {
    TCatBoostGrid::TCatBoostGrid(const TFullModel& model) {
        int nextInternalFeatureIdx = 0;
        for (const auto& floatFeature : model.ModelTrees->GetFloatFeatures()) {
            FeatureTypes.emplace_back(EFeatureType::Float);
            InternalToExternalFeature.emplace_back(floatFeature.Position.FlatIndex);
            FloatFeatureToInternalFeature[floatFeature.Position.Index] = nextInternalFeatureIdx;
            Borders.push_back(floatFeature.Borders);
            nextInternalFeatureIdx++;
        }

        const auto& catFeatures = model.ModelTrees->GetCatFeatures();
        for (const auto& oneHotFeature : model.ModelTrees->GetOneHotFeatures()) {
            FeatureTypes.emplace_back(EFeatureType::OneHot);
            InternalToExternalFeature.emplace_back(
                catFeatures[oneHotFeature.CatFeatureIndex].Position.FlatIndex);
            CatFeatureToInternalFeature[catFeatures[oneHotFeature.CatFeatureIndex].Position.Index] = nextInternalFeatureIdx;
            Borders.emplace_back();
            for (auto value : oneHotFeature.Values) {
                Borders.back().push_back(value);
            }
            nextInternalFeatureIdx++;
        }

        // TODO: support other feature types
        CB_ENSURE(model.ModelTrees->GetCtrFeatures().empty(), "CTRs are not supported");
        CB_ENSURE(model.ModelTrees->GetTextFeatures().empty(), "TextFeatures are not supported");
        CB_ENSURE(model.ModelTrees->GetEstimatedFeatures().empty(),
                  "EstimatedFeatures are not supported");

        BorderToIdx.resize(FeatureCount());
        BorderIdxToBinFeature.resize(FeatureCount());
        for (auto featureIdx : xrange(Borders.size())) {
            BorderIdxToBinFeature[featureIdx].resize(BorderCount(featureIdx));
            for (auto borderIdx : xrange(Borders[featureIdx].size())) {
                BorderToIdx[featureIdx][Borders[featureIdx][borderIdx]] = borderIdx;
                BorderIdxToBinFeature[featureIdx][borderIdx] = BinFeatures.size();
                if (FeatureTypes[featureIdx] == EFeatureType::Float) {
                    BinFeatures.emplace_back(featureIdx, borderIdx, EBinSplitType::TakeGreater);
                } else {
                    BinFeatures.emplace_back(featureIdx, borderIdx, EBinSplitType::TakeBin);
                }
            }
        }
    }
}
