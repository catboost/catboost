#include "compare_documents.h"

#include <catboost/private/libs/algo/apply.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/data/data_provider.h>
#include <util/generic/utility.h>


using namespace NCB;

static void CalcIndicatorCoefficients(
    const TFullModel& model,
    const TVector<ui32>& borderIdxForSplit,
    TArrayRef<ui32> treeLeafIdxes,
    TVector<TVector<double>>* floatFeatureImpact
) {
    const auto treeSplitOffsets = model.ModelTrees->GetTreeStartOffsets();
    floatFeatureImpact->resize(model.ModelTrees->GetFloatFeatures().size());

    for (ui32 featureIdx = 0; featureIdx < floatFeatureImpact->size(); ++featureIdx) {
        (*floatFeatureImpact)[featureIdx].resize(model.ModelTrees->GetFloatFeatures()[featureIdx].Borders.size() + 1);
    }

    const auto& binSplits = model.ModelTrees->GetBinFeatures();

    size_t offset = 0;
    for (size_t treeId = 0; treeId < model.ModelTrees->GetTreeCount(); ++treeId) {
        double leafValue = model.ModelTrees->GetLeafValues()[offset + treeLeafIdxes[treeId]];
        const size_t treeDepth = model.ModelTrees->GetTreeSizes().at(treeId);
        for (size_t depthIdx = 0; depthIdx < treeDepth; ++depthIdx) {
            size_t leafIdx = treeLeafIdxes[treeId] ^ (1 << depthIdx);
            double diff = model.ModelTrees->GetLeafValues()[offset + leafIdx] - leafValue;
            if (abs(diff) < 1e-12) {
                continue;
            }
            const auto splitIdx = model.ModelTrees->GetTreeSplits()[treeSplitOffsets[treeId] + depthIdx];
            auto split = binSplits[splitIdx];
            Y_ASSERT(split.Type == ESplitType::FloatFeature);
            int featureIdx = split.FloatFeature.FloatFeature;
            auto splitNewIdx = borderIdxForSplit[splitIdx] + ((leafIdx >> depthIdx) & 1);
            (*floatFeatureImpact)[featureIdx][splitNewIdx] += diff;
        }
        offset += (1ull << treeDepth);
    }
}

static TVector<double> GetPredictionDiffSingle(
    const TFeaturesLayout& layout,
    const TFullModel& model,
    const TVector<ui32>& borderIdxForSplit,
    const TVector<ui32>& borders,
    TArrayRef<ui32> treeLeafIdxes,
    bool tryIncrease
) {
    TVector<TVector<double>> floatFeatureImpact;
    CalcIndicatorCoefficients(
        model,
        borderIdxForSplit,
        treeLeafIdxes,
        &floatFeatureImpact
    );
    TVector<double> impact(layout.GetExternalFeatureCount());

    for (const auto& feature: model.ModelTrees->GetFloatFeatures()) {
        int featureIdx = feature.Position.Index;
        double diff = 0;
        auto externalIdx = feature.Position.FlatIndex;
        for (int splitIdx = borders[featureIdx] - 1; splitIdx >= 0; --splitIdx) {
            diff += floatFeatureImpact[featureIdx][splitIdx];
            if ((tryIncrease && diff < 0.) || (!tryIncrease && diff > 0.)) {
                impact[externalIdx] = std::max(impact[externalIdx], abs(diff));
            }
        }
        diff = 0;
        for (int splitIdx = borders[featureIdx] + 1; splitIdx < floatFeatureImpact[featureIdx].ysize(); ++splitIdx) {
            diff += floatFeatureImpact[featureIdx][splitIdx];
            if ((tryIncrease && diff < 0.) || (!tryIncrease && diff > 0.)) {
                impact[externalIdx] = std::max(impact[externalIdx], abs(diff));
            }
        }
    }
    return impact;
}

TVector<TVector<double>> GetPredictionDiff(
    const TFullModel& model,
    const TDataProvider& dataProvider,
    NPar::TLocalExecutor* localExecutor
) {
    CB_ENSURE(model.IsOblivious(), "Is not supported for non symmetric trees");
    CB_ENSURE(model.ModelTrees->GetDimensionsCount() == 1,  "Is not supported for multiclass");
    CB_ENSURE(dataProvider.GetObjectCount() == 2, "PredictionDiff requires 2 documents for compare");
    CB_ENSURE(model.GetNumCatFeatures() == 0, "Model with categorical features are not supported ");

    TVector<ui32> leafIdxes = CalcLeafIndexesMulti(model, dataProvider.ObjectsData, 0, 0);

    const auto *const rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(dataProvider.ObjectsData.Get());

    const auto& binSplits = model.ModelTrees->GetBinFeatures();

    TVector<TVector<ui32>> docBorders(dataProvider.GetObjectCount());
    TVector<ui32> borderIdxForSplit(binSplits.size(), std::numeric_limits<ui32>::infinity());
    ui32 splitIdx = 0;
    for (const auto& feature : model.ModelTrees->GetFloatFeatures()) {
        TMaybeData<const TFloatValuesHolder*> featureData
            = rawObjectsData->GetFloatFeature(feature.Position.FlatIndex);

        if (const auto* arrayColumn = dynamic_cast<const TFloatArrayValuesHolder*>(*featureData)) {
            arrayColumn->GetData()->ParallelForEach(
                [&] (ui32 docId, float value) {
                    docBorders[docId].push_back(0);
                    for (const auto& border: feature.Borders) {
                        if (value > border) {
                            docBorders[docId].back()++;
                        }
                    }
                },
                localExecutor
            );
        } else {
            CB_ENSURE_INTERNAL(false, "GetPredictionDiff: Unsupported column type");
        }
        if (splitIdx == binSplits.size() ||
            binSplits[splitIdx].Type != ESplitType::FloatFeature ||
            binSplits[splitIdx].FloatFeature.FloatFeature > feature.Position.Index
        ) {
            continue;
        }
        Y_ASSERT(binSplits[splitIdx].FloatFeature.FloatFeature >= feature.Position.Index);
        for (ui32 idx = 0; idx < feature.Borders.size() && binSplits[splitIdx].FloatFeature.FloatFeature == feature.Position.Index; ++idx) {
            if (abs(binSplits[splitIdx].FloatFeature.Split - feature.Borders[idx]) < 1e-15) {
                borderIdxForSplit[splitIdx] = idx;
                ++splitIdx;
            }
        }
    }

    TVector<double> predict = ApplyModelMulti(
        model,
        *dataProvider.ObjectsData,
        EPredictionType::RawFormulaVal,
        0,
        0,
        localExecutor)[0];

    TVector<TVector<double>> impact(dataProvider.GetObjectCount());
    for (size_t idx = 0; idx < dataProvider.GetObjectCount(); ++idx) {
        impact[idx] = GetPredictionDiffSingle(
            *dataProvider.ObjectsData.Get()->GetFeaturesLayout(),
            model,
            borderIdxForSplit,
            docBorders[idx],
            MakeArrayRef(leafIdxes.data() + idx * model.GetTreeCount(), model.GetTreeCount()),
            predict[idx] - predict[idx ^ 1] > 0
        );
    }
    TVector<TVector<double>> result(impact[0].size());
    for (size_t featureIdx = 0; featureIdx < impact[0].size(); ++ featureIdx) {
        result[featureIdx].push_back({std::abs(impact[0][featureIdx] + impact[1][featureIdx])});
    }
    return result;
}

void CalcAndOutputPredictionDiff(
    const TFullModel& model,
    const NCB::TDataProvider& dataProvider,
    const TString& outputPath,
    NPar::TLocalExecutor* localExecutor
) {
    auto factorImpact = GetPredictionDiff(model, dataProvider, localExecutor);
    TVector<std::pair<double, int>> impacts;
    for (const auto& impact: factorImpact) {
        impacts.push_back({impact[0], impacts.size()});
    }
    Sort(impacts.begin(), impacts.end(), std::greater<std::pair<double, int>>());

    TFileOutput out(outputPath);
    for (const auto& impact: impacts) {
        out << impact.first << " " << impact.second << Endl;
    }
}
