#include "grid.h"

namespace NMonoForest {
    TCatBoostGrid::TCatBoostGrid(const TFullModel& model) {
        int nextInternalFeatureIdx = 0;
        for (const auto& floatFeature : model.ObliviousTrees->GetFloatFeatures()) {
            FeatureTypes.emplace_back(EFeatureType::Float);
            InternalToExternalFeature.emplace_back(floatFeature.Position.FlatIndex);
            FloatFeatureToInternalFeature[floatFeature.Position.Index] = nextInternalFeatureIdx;
            Borders.push_back(floatFeature.Borders);
            nextInternalFeatureIdx++;
        }

        const auto& catFeatures = model.ObliviousTrees->GetCatFeatures();
        for (const auto& oneHotFeature : model.ObliviousTrees->GetOneHotFeatures()) {
            FeatureTypes.emplace_back(EFeatureType::OneHot);
            InternalToExternalFeature.emplace_back(
                catFeatures[oneHotFeature.CatFeatureIndex].Position.FlatIndex);
            CatFeatureToInternalFeature[catFeatures[oneHotFeature.CatFeatureIndex].Position.FlatIndex] = nextInternalFeatureIdx;
            Borders.emplace_back();
            for (auto value : oneHotFeature.Values) {
                Borders.back().push_back(value);
            }
            nextInternalFeatureIdx++;
        }

        // TODO: support other feature types
        CB_ENSURE(model.ObliviousTrees->GetCtrFeatures().empty(), "CTRs are not supported");
        CB_ENSURE(model.ObliviousTrees->GetTextFeatures().empty(), "TextFeatures are not supported");
        CB_ENSURE(model.ObliviousTrees->GetEstimatedFeatures().empty(),
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
