#include "tree_statistics.h"
#include "ders_helpers.h"

#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/private/libs/options/json_helper.h>


using namespace NCB;


// ITreeStatisticsEvaluator

TVector<TTreeStatistics> ITreeStatisticsEvaluator::EvaluateTreeStatistics(
    const TFullModel& model,
    const NCB::TProcessedDataProvider& processedData,
    const TMaybe<double> startingApprox,
    int logPeriod
) {
    //TODO(eermishkina): support non symmetric trees
    CB_ENSURE_INTERNAL(model.IsOblivious(), "Is supported only for symmetric trees");

    NJson::TJsonValue paramsJson = ReadTJsonValue(model.ModelInfo.at("params"));
    const ELossFunction lossFunction = FromString<ELossFunction>(paramsJson["loss_function"]["type"].GetString());
    const ELeavesEstimation leafEstimationMethod = FromString<ELeavesEstimation>(paramsJson["tree_learner_options"]["leaf_estimation_method"].GetString());
    const ui32 leavesEstimationIterations = paramsJson["tree_learner_options"]["leaf_estimation_iterations"].GetUInteger();
    const float learningRate = paramsJson["boosting_options"]["learning_rate"].GetDouble();
    const float l2LeafReg = paramsJson["tree_learner_options"]["l2_leaf_reg"].GetDouble();
    const ui32 treeCount = model.GetTreeCount();

    auto binarizedFeatures = MakeQuantizedFeaturesForEvaluator(model, *processedData.ObjectsData.Get());
    TVector<TTreeStatistics> treeStatistics;
    treeStatistics.reserve(treeCount);
    TVector<double> approxes(DocCount, startingApprox ? *startingApprox : 0);

    TImportanceLogger treesLogger(treeCount, "Trees processed", "Processing trees...", logPeriod);
    TProfileInfo processTreesProfile(treeCount);

    for (ui32 treeId = 0; treeId < treeCount; ++treeId) {
        processTreesProfile.StartIterationBlock();

        LeafCount = 1 << model.ModelTrees->GetModelTreeData()->GetTreeSizes()[treeId];
        LeafIndices = BuildIndicesForBinTree(model, binarizedFeatures.Get(), treeId);

        TVector<TVector<ui32>> leavesDocId(LeafCount);
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leavesDocId[LeafIndices[docId]].push_back(docId);
        }


        TVector<TVector<double>> leafValues(leavesEstimationIterations);
        TVector<TVector<double>> formulaDenominators(leavesEstimationIterations);
        TVector<TVector<double>> formulaNumeratorAdding(leavesEstimationIterations);
        TVector<TVector<double>> formulaNumeratorMultiplier(leavesEstimationIterations);
        TVector<double> localApproxes(approxes);

        TConstArrayRef<float> weights = GetWeights(*processedData.TargetData);

        for (ui32 it = 0; it < leavesEstimationIterations; ++it) {
            EvaluateDerivatives(
                lossFunction,
                leafEstimationMethod,
                localApproxes,
                *processedData.TargetData->GetOneDimensionalTarget(),
                &FirstDerivatives,
                &SecondDerivatives,
                &ThirdDerivatives
            );

            const TVector<double> leafNumerators = ComputeLeafNumerators(weights);
            TVector<double> leafDenominators = ComputeLeafDenominators(weights, l2LeafReg);
            LeafValues.resize(LeafCount);
            for (ui32 leafId = 0; leafId < LeafCount; ++leafId) {
                LeafValues[leafId] = -leafNumerators[leafId] / leafDenominators[leafId];
            }
            formulaNumeratorAdding[it] = ComputeFormulaNumeratorAdding();
            formulaNumeratorMultiplier[it] = ComputeFormulaNumeratorMultiplier(weights);
            formulaDenominators[it].swap(leafDenominators);

            for (ui32 docId = 0; docId < DocCount; ++docId) {
                localApproxes[docId] += LeafValues[LeafIndices[docId]];
            }
            leafValues[it].swap(LeafValues);
        }

        for (auto& leafValuesOneIteration : leafValues) {
            for (auto& leafValue : leafValuesOneIteration) {
                leafValue *= learningRate;
            }
            for (ui32 docId = 0; docId < DocCount; ++docId) {
                approxes[docId] += leafValuesOneIteration[LeafIndices[docId]];
            }
        }

        treeStatistics.push_back({
            LeafCount,
            LeafIndices,
            leavesDocId,
            leafValues,
            formulaDenominators,
            formulaNumeratorAdding,
            formulaNumeratorMultiplier
        });

        processTreesProfile.FinishIteration();
        auto profileResults = processTreesProfile.GetProfileResults();
        treesLogger.Log(profileResults);
    }
    return treeStatistics;
}

// TGradientTreeStatisticsEvaluator

TVector<double> TGradientTreeStatisticsEvaluator::ComputeLeafNumerators(TConstArrayRef<float> weights) {
    TVector<double> leafNumerators(LeafCount);
    if (weights.empty()) {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafNumerators[LeafIndices[docId]] += FirstDerivatives[docId];
        }
    } else {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafNumerators[LeafIndices[docId]] += weights[docId] * FirstDerivatives[docId];
        }
    }
    return leafNumerators;
}

TVector<double> TGradientTreeStatisticsEvaluator::ComputeLeafDenominators(TConstArrayRef<float> weights, float l2LeafReg) {
    TVector<double> leafDenominators(LeafCount);
    if (weights.empty()) {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafDenominators[LeafIndices[docId]] += 1;
        }
    } else {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafDenominators[LeafIndices[docId]] += weights[docId];
        }
    }
    for (ui32 leafId = 0; leafId < LeafCount; ++leafId) {
        leafDenominators[leafId] += l2LeafReg;
    }
    return leafDenominators;
}

TVector<double> TGradientTreeStatisticsEvaluator::ComputeFormulaNumeratorAdding() {
    TVector<double> formulaNumeratorAdding(DocCount);
    for (ui32 docId = 0; docId < DocCount; ++docId) {
        formulaNumeratorAdding[docId] = LeafValues[LeafIndices[docId]] + FirstDerivatives[docId];
    }
    return formulaNumeratorAdding;
}

TVector<double> TGradientTreeStatisticsEvaluator::ComputeFormulaNumeratorMultiplier(TConstArrayRef<float> weights) {
    TVector<double> formulaNumeratorMultiplier(DocCount);
    if (weights.empty()) {
        formulaNumeratorMultiplier = SecondDerivatives;
    } else {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            formulaNumeratorMultiplier[docId] = weights[docId] * SecondDerivatives[docId];
        }
    }
    return formulaNumeratorMultiplier;
}

// TNewtonTreeStatisticsEvaluator

TVector<double> TNewtonTreeStatisticsEvaluator::ComputeLeafNumerators(TConstArrayRef<float> weights) {
    TVector<double> leafNumerators(LeafCount);
    if (weights.empty()) {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafNumerators[LeafIndices[docId]] += FirstDerivatives[docId];
        }
    } else {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafNumerators[LeafIndices[docId]] += weights[docId] * FirstDerivatives[docId];
        }
    }
    return leafNumerators;
}

TVector<double> TNewtonTreeStatisticsEvaluator::ComputeLeafDenominators(TConstArrayRef<float> weights, float l2LeafReg) {
    TVector<double> leafDenominators(LeafCount);
    if (weights.empty()) {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafDenominators[LeafIndices[docId]] += SecondDerivatives[docId];
        }
    } else {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            leafDenominators[LeafIndices[docId]] += weights[docId] * SecondDerivatives[docId];
        }
    }
    for (ui32 leafId = 0; leafId < LeafCount; ++leafId) {
        leafDenominators[leafId] += l2LeafReg;
    }
    return leafDenominators;
}

TVector<double> TNewtonTreeStatisticsEvaluator::ComputeFormulaNumeratorAdding() {
    TVector<double> formulaNumeratorAdding(DocCount);
    for (ui32 docId = 0; docId < DocCount; ++docId) {
        formulaNumeratorAdding[docId] = LeafValues[LeafIndices[docId]] * SecondDerivatives[docId] + FirstDerivatives[docId];
    }
    return formulaNumeratorAdding;
}

TVector<double> TNewtonTreeStatisticsEvaluator::ComputeFormulaNumeratorMultiplier(TConstArrayRef<float> weights) {
    TVector<double> formulaNumeratorMultiplier(DocCount);
    if (weights.empty()) {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            formulaNumeratorMultiplier[docId] = LeafValues[LeafIndices[docId]] * ThirdDerivatives[docId] + SecondDerivatives[docId];
        }
    } else {
        for (ui32 docId = 0; docId < DocCount; ++docId) {
            formulaNumeratorMultiplier[docId] = weights[docId] * (LeafValues[LeafIndices[docId]] * ThirdDerivatives[docId] + SecondDerivatives[docId]);
        }
    }
    return formulaNumeratorMultiplier;
}
