#include "util.h"

#include <catboost/private/libs/algo/features_data_helpers.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/model/cpu/evaluator.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/target/data_providers.h>

#include <util/generic/mapfindptr.h>


using namespace NCB;


TVector<double> CollectLeavesStatistics(
    const NCB::TDataProvider& dataset,
    const TFullModel& model,
    NPar::TLocalExecutor* localExecutor) {

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
                localExecutor).TargetData;

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
        model.ModelTrees->GetLeafValues().size() / approxDimension
    );

    auto binFeatures = MakeQuantizedFeaturesForEvaluator(model, *dataset.ObjectsData.Get());

    const auto documentsCount = dataset.GetObjectCount();
    for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        TVector<TIndexType> indices = BuildIndicesForBinTree(model, binFeatures.Get(), treeIdx);
        const int offset = model.ModelTrees->GetFirstLeafOffsets()[treeIdx] / approxDimension;
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

bool TryGetLossDescription(const TFullModel& model, NCatboostOptions::TLossDescription& lossDescription) {
    if (!(model.ModelInfo.contains("loss_function") ||  model.ModelInfo.contains("params") && ReadTJsonValue(model.ModelInfo.at("params")).Has("loss_function"))) {
        return false;
    }
    if (model.ModelInfo.contains("loss_function")) {
        lossDescription.Load(ReadTJsonValue(model.ModelInfo.at("loss_function")));
    } else {
        lossDescription.Load(ReadTJsonValue(model.ModelInfo.at("params"))["loss_function"]);
    }
    return true;
}

void CheckNonZeroApproxForZeroWeightLeaf(const TFullModel& model) {
    for (size_t leafIdx = 0; leafIdx < model.ModelTrees->GetLeafWeights().size(); ++leafIdx) {
        size_t approxDimension = model.GetDimensionsCount();
        if (model.ModelTrees->GetLeafWeights()[leafIdx] == 0) {
            double leafSumApprox = 0;
            for (size_t approxIdx = 0; approxIdx < approxDimension; ++approxIdx) {
                leafSumApprox += abs(model.ModelTrees->GetLeafValues()[leafIdx * approxDimension + approxIdx]);
            }
            CB_ENSURE(leafSumApprox < 1e-9, "Cannot calc shap values, model contains non zero approx for zero-weight leaf");
        }
    }
}
