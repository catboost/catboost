#include "compare_documents.h"

#include <catboost/private/libs/algo/apply.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/data/data_provider.h>
#include <util/generic/utility.h>


using namespace NCB;

static void CalcObliviousIndicatorCoefficients(
    const TFullModel& model,
    const TVector<ui32>& borderIdxForSplit,
    TArrayRef<ui32> treeLeafIdxes,
    TVector<TVector<double>>* floatFeatureImpact
) {
    const auto& treeSplitOffsets = model.ModelTrees->GetModelTreeData()->GetTreeStartOffsets();
    floatFeatureImpact->resize(model.ModelTrees->GetFloatFeatures().size());

    for (ui32 featureIdx = 0; featureIdx < floatFeatureImpact->size(); ++featureIdx) {
        (*floatFeatureImpact)[featureIdx].resize(model.ModelTrees->GetFloatFeatures()[featureIdx].Borders.size() + 1);
    }

    const auto& binSplits = model.ModelTrees->GetBinFeatures();

    size_t offset = 0;
    for (size_t treeId = 0; treeId < model.ModelTrees->GetTreeCount(); ++treeId) {
        double leafValue = model.ModelTrees->GetModelTreeData()->GetLeafValues()[offset + treeLeafIdxes[treeId]];
        const size_t treeDepth = model.ModelTrees->GetModelTreeData()->GetTreeSizes().at(treeId);
        for (size_t depthIdx = 0; depthIdx < treeDepth; ++depthIdx) {
            size_t leafIdx = treeLeafIdxes[treeId] ^ (1 << depthIdx);
            double diff = model.ModelTrees->GetModelTreeData()->GetLeafValues()[offset + leafIdx] - leafValue;
            if (abs(diff) < 1e-12) {
                continue;
            }
            const int splitIdx = model.ModelTrees->GetModelTreeData()->GetTreeSplits()[treeSplitOffsets[treeId] + depthIdx];
            auto split = binSplits[splitIdx];
            CB_ENSURE_INTERNAL(split.Type == ESplitType::FloatFeature, "Models only with float features are supported");
            int featureIdx = split.FloatFeature.FloatFeature;
            ui32 splitNewIdx = borderIdxForSplit[splitIdx] + ((leafIdx >> depthIdx) & 1);
            (*floatFeatureImpact)[featureIdx][splitNewIdx] += diff;
        }
        offset += (1ull << treeDepth);
    }
}

static double CalcNonSymmetricLeafValue(
    const TFullModel& model,
    int splitIdx,
    const TVector<double>& featureValues
) {
    const auto& treeSplits = model.ModelTrees->GetModelTreeData()->GetTreeSplits();
    const auto& stepNodes = model.ModelTrees->GetModelTreeData()->GetNonSymmetricStepNodes();
    const auto& binFeatures = model.ModelTrees->GetBinFeatures();

    int nextNodeStep = 0;
    do {
        const auto& split = binFeatures[treeSplits[splitIdx]];
        double featureValue = featureValues[split.FloatFeature.FloatFeature];
        double splitValue = split.FloatFeature.Split;
        nextNodeStep = (featureValue > splitValue) ? stepNodes[splitIdx].RightSubtreeDiff : stepNodes[splitIdx].LeftSubtreeDiff;
        splitIdx += nextNodeStep;
    } while (nextNodeStep != 0);
    ui32 leafIdx = model.ModelTrees->GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[splitIdx];
    return model.ModelTrees->GetModelTreeData()->GetLeafValues()[leafIdx];
}


static void CalcNonSymmetricIndicatorCoefficients(
        const TFullModel& model,
        const TVector<ui32>& borderIdxForSplit,
        const TVector<double>& featureValues,
        TArrayRef<ui32> treeLeafIdxes,
        TVector<TVector<double>>* floatFeatureImpact
) {
    floatFeatureImpact->resize(model.ModelTrees->GetFloatFeatures().size());
    for (ui32 featureIdx = 0; featureIdx < floatFeatureImpact->size(); ++featureIdx) {
        (*floatFeatureImpact)[featureIdx].resize(model.ModelTrees->GetFloatFeatures()[featureIdx].Borders.size() + 1);
    }

    const auto& treeSplits = model.ModelTrees->GetModelTreeData()->GetTreeSplits();
    const auto& stepNodes = model.ModelTrees->GetModelTreeData()->GetNonSymmetricStepNodes();
    const auto& binFeatures = model.ModelTrees->GetBinFeatures();

    auto applyData = model.ModelTrees->GetApplyData();
    for (size_t treeId = 0; treeId < model.ModelTrees->GetTreeCount(); ++treeId) {
        size_t leafOffset = applyData->TreeFirstLeafOffsets[treeId];
        double leafValue = model.ModelTrees->GetModelTreeData()->GetLeafValues()[leafOffset + treeLeafIdxes[treeId]];
        size_t splitIdx = model.ModelTrees->GetModelTreeData()->GetTreeStartOffsets()[treeId];
        int nextSplitStep = 0;
        do {
            const auto& split = binFeatures[treeSplits[splitIdx]];
            double featureValue = featureValues[split.FloatFeature.FloatFeature];
            double splitValue = split.FloatFeature.Split;
            int newSplitStep = featureValue > splitValue ? stepNodes[splitIdx].LeftSubtreeDiff : stepNodes[splitIdx].RightSubtreeDiff;
            size_t newSplitIdx = splitIdx + newSplitStep;
            double newLeafValue = CalcNonSymmetricLeafValue(model, newSplitIdx, featureValues);
            double diff = newLeafValue - leafValue;
            if (abs(diff) > 1e-12) {
                ui32 borderIdx = borderIdxForSplit[treeSplits[splitIdx]] + (featureValue < splitValue);
                (*floatFeatureImpact)[split.FloatFeature.FloatFeature][borderIdx] += diff;
            }
            nextSplitStep = featureValue > splitValue ? stepNodes[splitIdx].RightSubtreeDiff : stepNodes[splitIdx].LeftSubtreeDiff;
            splitIdx += nextSplitStep;
        } while (nextSplitStep != 0);
    }

}

static void CalcIndicatorCoefficients(
        const TFullModel& model,
        const TVector<ui32>& borderIdxForSplit,
        const TVector<double>& featureValues,
        TArrayRef<ui32> treeLeafIdxes,
        TVector<TVector<double>>* floatFeatureImpact
) {
 if (model.IsOblivious()) {
     CalcObliviousIndicatorCoefficients(model, borderIdxForSplit, treeLeafIdxes, floatFeatureImpact);
 } else {
     CalcNonSymmetricIndicatorCoefficients(model, borderIdxForSplit, featureValues, treeLeafIdxes, floatFeatureImpact);
 }
}

static TVector<double> GetPredictionDiffSingle(
    const TFeaturesLayout& layout,
    const TVector<double>& featureValues,
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
        featureValues,
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
            if ((tryIncrease && diff > 0.) || (!tryIncrease && diff < 0.)) {
                impact[externalIdx] = std::max(impact[externalIdx], abs(diff));
            }
        }
        diff = 0;
        for (int splitIdx = borders[featureIdx] + 1; splitIdx < floatFeatureImpact[featureIdx].ysize(); ++splitIdx) {
            diff += floatFeatureImpact[featureIdx][splitIdx];
            if ((tryIncrease && diff > 0.) || (!tryIncrease && diff < 0.)) {
                impact[externalIdx] = std::max(impact[externalIdx], abs(diff));
            }
        }
    }
    return impact;
}

TVector<double> GetPredictionDiff(
    const TFullModel& model,
    const TObjectsDataProviderPtr objectsDataProvider,
    NPar::ILocalExecutor* localExecutor
) {
    CB_ENSURE(model.ModelTrees->GetDimensionsCount() == 1,  "Is not supported for multiclass");
    CB_ENSURE(objectsDataProvider->GetObjectCount() == 2, "PredictionDiff requires 2 documents for compare");
    CB_ENSURE(model.GetNumCatFeatures() == 0, "Models with categorical features are not supported");

    TVector<ui32> leafIdxes = CalcLeafIndexesMulti(model, objectsDataProvider, 0, 0);

    const auto* const rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(objectsDataProvider.Get());

    const auto& binSplits = model.ModelTrees->GetBinFeatures();

    TVector<TVector<double>> floatFeatureValues(objectsDataProvider->GetObjectCount());
    for (size_t idx = 0; idx < objectsDataProvider->GetObjectCount(); ++idx) {
        floatFeatureValues[idx].resize(model.GetNumFloatFeatures());
    }

    TVector<TVector<ui32>> docBorders(objectsDataProvider->GetObjectCount());
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
                            floatFeatureValues[docId][feature.Position.FlatIndex] = value;
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

    // TODO : Support baselines
    TVector<double> predict = ApplyModelMulti(
        model,
        *objectsDataProvider,
        EPredictionType::RawFormulaVal,
        0,
        0,
        localExecutor)[0];

    TVector<TVector<double>> impact(objectsDataProvider->GetObjectCount());
    for (size_t idx = 0; idx < objectsDataProvider->GetObjectCount(); ++idx) {
        impact[idx] = GetPredictionDiffSingle(
            *objectsDataProvider->GetFeaturesLayout(),
            floatFeatureValues[idx],
            model,
            borderIdxForSplit,
            docBorders[idx],
            MakeArrayRef(leafIdxes.data() + idx * model.GetTreeCount(), model.GetTreeCount()),
            predict[idx] - predict[idx ^ 1] < 0
        );
    }
    TVector<double> result;
    for (size_t featureIdx = 0; featureIdx < impact[0].size(); ++ featureIdx) {
        result.push_back(std::abs(impact[0][featureIdx] + impact[1][featureIdx]));
    }
    return result;
}

TVector<TVector<double>> GetPredictionDiff(
    const TFullModel& model,
    const TDataProvider& dataProvider,
    NPar::ILocalExecutor* localExecutor
) {
    TVector<double> result1d = GetPredictionDiff(model, dataProvider.ObjectsData, localExecutor);
    TVector<TVector<double>> result(result1d.size());
    for (size_t featureIdx = 0; featureIdx < result1d.size(); ++ featureIdx) {
        result[featureIdx].push_back({result1d[featureIdx]});
    }
    return result;
}

void CalcAndOutputPredictionDiff(
    const TFullModel& model,
    const NCB::TDataProvider& dataProvider,
    const TString& outputPath,
    NPar::ILocalExecutor* localExecutor
) {
    auto factorImpact = GetPredictionDiff(model, dataProvider, localExecutor);
    TVector<std::pair<double, int>> impacts;
    for (const auto& impact: factorImpact) {
        impacts.push_back({impact[0], impacts.size()});
    }
    StableSort(impacts.begin(), impacts.end(), std::greater<std::pair<double, int>>());

    TFileOutput out(outputPath);
    for (const auto& impact: impacts) {
        out << impact.first << " " << impact.second << Endl;
    }
}
