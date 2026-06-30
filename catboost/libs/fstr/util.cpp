#include "util.h"

#include <catboost/private/libs/algo/features_data_helpers.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/model/cpu/evaluator.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_estimated_features.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/target/data_providers.h>

#include <util/generic/mapfindptr.h>


using namespace NCB;


TVector<double> CollectLeavesStatistics(
    const TDataProvider& dataset,
    const TFullModel& model,
    NPar::ILocalExecutor* localExecutor) {

    TConstArrayRef<float> weights;

    TTargetDataProviderPtr targetData; // needed to own weights data

    if (const auto* modelInfoParams = MapFindPtr(model.ModelInfo, "params")) {
        NJson::TJsonValue paramsJson = ReadTJsonValue(*modelInfoParams);
        if (paramsJson.Has("loss_function")) {
            TRestorableFastRng64 rand(0);

            targetData = CreateModelCompatibleProcessedDataProvider(
                dataset,
                {},
                model,
                GetMonopolisticFreeCpuRam(),
                &rand,
                localExecutor,
                /*metricsThatRequireTargetCanBeSkipped*/true,
                /*skipMinMaxPairsCheck*/true,
                /*skipTargetConsistencyCheck*/true
            ).TargetData;

            weights = GetWeights(*targetData);
        }
    }

    // If it is impossible to get properly adjusted weights use raw weights from RawTargetData
    if (weights.empty()) {
        const TWeights<float>& rawWeights = dataset.RawTargetData.GetWeights();
        if (!rawWeights.IsTrivial()) {
            weights = rawWeights.GetNonTrivialData();
        }
    }

    size_t treeCount = model.GetTreeCount();
    const int approxDimension = model.ModelTrees->GetDimensionsCount();
    TVector<double> leavesStatistics(
        model.ModelTrees->GetModelTreeData()->GetLeafValues().size() / approxDimension
    );

    auto binFeatures = MakeQuantizedFeaturesForEvaluator(model, *dataset.ObjectsData.Get());

    const auto documentsCount = dataset.GetObjectCount();
    auto applyData = model.ModelTrees->GetApplyData();
    for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        TVector<TIndexType> indices = BuildIndicesForBinTree(model, binFeatures.Get(), treeIdx);
        const int offset = applyData->TreeFirstLeafOffsets[treeIdx] / approxDimension;
        if (indices.empty()) {
            continue;
        }

        if (weights.empty()) {
            for (size_t doc = 0; doc < documentsCount; ++doc) {
                const TIndexType valueIndex = indices[doc];
                leavesStatistics[offset + valueIndex] += 1.0;
            }
        } else {
            for (size_t doc = 0; doc < documentsCount; ++doc) {
                const TIndexType valueIndex = indices[doc];
                leavesStatistics[offset + valueIndex] += weights[doc];
            }
        }
    }
    return leavesStatistics;
}

bool TryGetLossDescription(const TFullModel& model, NCatboostOptions::TLossDescription* lossDescription) {
    if (!(model.ModelInfo.contains("loss_function") ||  model.ModelInfo.contains("params") && ReadTJsonValue(model.ModelInfo.at("params")).Has("loss_function"))) {
        return false;
    }
    if (model.ModelInfo.contains("loss_function")) {
        lossDescription->Load(ReadTJsonValue(model.ModelInfo.at("loss_function")));
    } else {
        lossDescription->Load(ReadTJsonValue(model.ModelInfo.at("params"))["loss_function"]);
    }
    return true;
}

bool TryGetObjectiveMetric(const TFullModel& model, NCatboostOptions::TLossDescription* lossDescription) {
    if (model.ModelInfo.contains("params")) {
        const auto &params = ReadTJsonValue(model.ModelInfo.at("params"));
        if (params.Has("metrics") && params["metrics"].Has("objective_metric")) {
            lossDescription->Load(params["metrics"]["objective_metric"]);
            return true;
        }
    }
    return TryGetLossDescription(model, lossDescription);
}

bool HasNonZeroApproxForZeroWeightLeaf(const TFullModel& model) {
    for (size_t leafIdx = 0; leafIdx < model.ModelTrees->GetModelTreeData()->GetLeafWeights().size(); ++leafIdx) {
        size_t approxDimension = model.GetDimensionsCount();
        if (model.ModelTrees->GetModelTreeData()->GetLeafWeights()[leafIdx] == 0) {
            double leafSumApprox = 0;
            for (size_t approxIdx = 0; approxIdx < approxDimension; ++approxIdx) {
                leafSumApprox += abs(model.ModelTrees->GetModelTreeData()->GetLeafValues()[leafIdx * approxDimension + approxIdx]);
            }
            if (leafSumApprox >= 1e-9) {
                return true;
            }
        }
    }
    return false;
}

bool NeedDatasetForLeavesWeights(const TFullModel& model, bool fstrOnTrainPool) {
    const auto leafWeightsOfModels = model.ModelTrees->GetModelTreeData()->GetLeafWeights();
    const bool needSumModelAndDatasetWeights = !fstrOnTrainPool && HasNonZeroApproxForZeroWeightLeaf(model);
    const bool output = leafWeightsOfModels.empty() || needSumModelAndDatasetWeights;
    return output;
}

TVector<int> GetBinFeatureCombinationClassByDepth(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    size_t treeIdx
) {
    const size_t depthOfTree = forest.GetModelTreeData()->GetTreeSizes()[treeIdx];
    TVector<int> binFeatureCombinationClassByDepth(depthOfTree);
    for (size_t depth = 0; depth < depthOfTree; ++depth) {
        const size_t remainingDepth = depthOfTree - depth - 1;
        const int combinationClass = binFeatureCombinationClass[
            forest.GetModelTreeData()->GetTreeSplits()[forest.GetModelTreeData()->GetTreeStartOffsets()[treeIdx] + remainingDepth]
        ];
        binFeatureCombinationClassByDepth[depth] = combinationClass;
    }
    return binFeatureCombinationClassByDepth;
}

EFeatureCalcerType GetEstimatedFeatureCalcerType(
    const TFullModel& model,
    const TModelEstimatedFeature& estimatedFeature
) {
    if (estimatedFeature.SourceFeatureType == EEstimatedSourceFeatureType::Text) {
        return model.TextProcessingCollection->GetCalcer(estimatedFeature.CalcerId)->Type();
    } else {
        CB_ENSURE_INTERNAL(
            estimatedFeature.SourceFeatureType == EEstimatedSourceFeatureType::Embedding,
            "Unexpected EEstimatedSourceFeatureType: " << estimatedFeature.SourceFeatureType
        );
        return model.EmbeddingProcessingCollection->GetCalcer(estimatedFeature.CalcerId)->Type();
    }
}
