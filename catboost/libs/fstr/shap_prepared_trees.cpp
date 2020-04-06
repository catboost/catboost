#include "shap_prepared_trees.h"

#include "shap_values.h"
#include "util.h"

#include <catboost/private/libs/algo/features_data_helpers.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/model/cpu/quantization.h>
#include <catboost/private/libs/options/restrictions.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/utility.h>
#include <util/generic/ymath.h>


using namespace NCB;


static TVector<double> CalcMeanValueForTree(
    const TModelTrees& forest,
    const TVector<TVector<double>>& subtreeWeights,
    size_t treeIdx
) {
    const int approxDimension = forest.GetDimensionsCount();
    TVector<double> meanValue(approxDimension, 0.0);

    if (forest.IsOblivious()) {
        auto firstLeafPtr = forest.GetFirstLeafPtrForTree(treeIdx);
        const size_t maxDepth = forest.GetTreeSizes()[treeIdx];
        for (size_t leafIdx = 0; leafIdx < (size_t(1) << maxDepth); ++leafIdx) {
            for (int dimension = 0; dimension < approxDimension; ++dimension) {
                meanValue[dimension] += firstLeafPtr[leafIdx * approxDimension + dimension]
                                     * subtreeWeights[maxDepth][leafIdx];
            }
        }
    } else {
        const int totalNodesCount = forest.GetNonSymmetricNodeIdToLeafId().size();
        const bool isLastTree = treeIdx == forest.GetTreeStartOffsets().size() - 1;
        const size_t startOffset = forest.GetTreeStartOffsets()[treeIdx];
        const size_t endOffset = isLastTree ? totalNodesCount : forest.GetTreeStartOffsets()[treeIdx + 1];
        for (size_t nodeIdx = startOffset; nodeIdx < endOffset; ++nodeIdx) {
            for (int dimension = 0; dimension < approxDimension; ++dimension) {
                size_t leafIdx = forest.GetNonSymmetricNodeIdToLeafId()[nodeIdx];
                if (leafIdx < forest.GetLeafValues().size()) {
                    meanValue[dimension] += forest.GetLeafValues()[leafIdx + dimension]
                                        * forest.GetLeafWeights()[leafIdx / forest.GetDimensionsCount()];
                }
            }
        }
    }

    for (int dimension = 0; dimension < approxDimension; ++dimension) {
        meanValue[dimension] /= subtreeWeights[0][0];
    }

    return meanValue;
}

// 'reversed' mean every child will become parent and vice versa
static TVector<size_t> GetReversedSubtreeForNonObliviousTree(
    const TModelTrees& forest,
    int treeIdx
) {
    const int totalNodesCount = forest.GetTreeSplits().size();
    const bool isLastTree = static_cast<size_t>(treeIdx + 1) == forest.GetTreeStartOffsets().size();
    const int startOffset = forest.GetTreeStartOffsets()[treeIdx];
    const int endOffset = isLastTree ? totalNodesCount : forest.GetTreeStartOffsets()[treeIdx + 1];
    const int treeSize = endOffset - startOffset;

    TVector<size_t> reversedTree(treeSize, 0);
    for (int nodeIdx = startOffset; nodeIdx < endOffset; ++nodeIdx) {
        const int localIdx = nodeIdx - startOffset;
        const size_t leftDiff = forest.GetNonSymmetricStepNodes()[nodeIdx].LeftSubtreeDiff;
        const size_t rightDiff = forest.GetNonSymmetricStepNodes()[nodeIdx].RightSubtreeDiff;
        if (leftDiff != 0) {
            reversedTree[localIdx + leftDiff] = nodeIdx;
        }
        if (rightDiff != 0) {
            reversedTree[localIdx + rightDiff] = nodeIdx;
        }
    }
    return reversedTree;
}

static TVector<TVector<double>> CalcSubtreeWeightsForTree(
    const TModelTrees& forest,
    const TVector<double>& leafWeights,
    int treeIdx
) {
    TVector<TVector<double>> subtreeWeights;
    if (forest.IsOblivious()) {
        const int treeDepth = forest.GetTreeSizes()[treeIdx];
        subtreeWeights.resize(treeDepth + 1);
        subtreeWeights[treeDepth].resize(size_t(1) << treeDepth);
        const int weightOffset = forest.GetFirstLeafOffsets()[treeIdx] / forest.GetDimensionsCount();

        for (size_t nodeIdx = 0; nodeIdx < size_t(1) << treeDepth; ++nodeIdx) {
            subtreeWeights[treeDepth][nodeIdx] = leafWeights[weightOffset + nodeIdx];
        }

        for (int depth = treeDepth - 1; depth >= 0; --depth) {
            const size_t nodeCount = size_t(1) << depth;
            subtreeWeights[depth].resize(nodeCount);
            for (size_t nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx) {
                subtreeWeights[depth][nodeIdx] = subtreeWeights[depth + 1][nodeIdx * 2] + subtreeWeights[depth + 1][nodeIdx * 2 + 1];
            }
        }
    } else {
        const int startOffset = forest.GetTreeStartOffsets()[treeIdx];
        TVector<size_t> reversedTree = GetReversedSubtreeForNonObliviousTree(forest, treeIdx);
        subtreeWeights.resize(1); // with respect to NonSymmetric format of TObliviousTree
        subtreeWeights[0].resize(reversedTree.size(), 0);
        if (reversedTree.size() == 1) {
            subtreeWeights[0][0] = leafWeights[forest.GetNonSymmetricNodeIdToLeafId()[startOffset] / forest.GetDimensionsCount()];
        } else {
            for (size_t localIdx = reversedTree.size() - 1; localIdx > 0; --localIdx) {
                size_t leafIdx = forest.GetNonSymmetricNodeIdToLeafId()[startOffset + localIdx] / forest.GetDimensionsCount();
                if (leafIdx < leafWeights.size()) {
                    subtreeWeights[0][localIdx] += leafWeights[leafIdx];
                }
                subtreeWeights[0][reversedTree[localIdx] - startOffset] += subtreeWeights[0][localIdx];
            }
        }
    }
    return subtreeWeights;
}

static void MapBinFeaturesToClasses(
    const TModelTrees& forest,
    TVector<int>* binFeatureCombinationClass,
    TVector<TVector<int>>* combinationClassFeatures
) {
    TConstArrayRef<TFloatFeature> floatFeatures = forest.GetFloatFeatures();
    TConstArrayRef<TCatFeature> catFeatures = forest.GetCatFeatures();
    const NCB::TFeaturesLayout layout(
        TVector<TFloatFeature>(floatFeatures.begin(), floatFeatures.end()),
        TVector<TCatFeature>(catFeatures.begin(), catFeatures.end()));
    TVector<TVector<int>> featuresCombinations;
    TVector<size_t> featureBucketSizes;

    for (const TFloatFeature& floatFeature : forest.GetFloatFeatures()) {
        if (!floatFeature.UsedInModel()) {
            continue;
        }
        featuresCombinations.emplace_back();
        featuresCombinations.back() = { floatFeature.Position.FlatIndex };
        featureBucketSizes.push_back(floatFeature.Borders.size());
    }

    for (const TOneHotFeature& oneHotFeature: forest.GetOneHotFeatures()) {
        featuresCombinations.emplace_back();
        featuresCombinations.back() = {
            (int)layout.GetExternalFeatureIdx(oneHotFeature.CatFeatureIndex,
            EFeatureType::Categorical)
        };
        featureBucketSizes.push_back(oneHotFeature.Values.size());
    }

    for (const TCtrFeature& ctrFeature : forest.GetCtrFeatures()) {
        const TFeatureCombination& combination = ctrFeature.Ctr.Base.Projection;
        featuresCombinations.emplace_back();
        for (int catFeatureIdx : combination.CatFeatures) {
            featuresCombinations.back().push_back(
                layout.GetExternalFeatureIdx(catFeatureIdx, EFeatureType::Categorical));
        }
        featureBucketSizes.push_back(ctrFeature.Borders.size());
    }
    TVector<size_t> featureFirstBinBucket(featureBucketSizes.size(), 0);
    for (size_t i = 1; i < featureBucketSizes.size(); ++i) {
        featureFirstBinBucket[i] = featureFirstBinBucket[i - 1] + featureBucketSizes[i - 1];
    }
    TVector<int> sortedBinFeatures(featuresCombinations.size());
    Iota(sortedBinFeatures.begin(), sortedBinFeatures.end(), 0);
    Sort(
        sortedBinFeatures.begin(),
        sortedBinFeatures.end(),
        [featuresCombinations](int feature1, int feature2) {
            return featuresCombinations[feature1] < featuresCombinations[feature2];
        }
    );

    *binFeatureCombinationClass = TVector<int>(forest.GetBinaryFeaturesFullCount());
    *combinationClassFeatures = TVector<TVector<int>>();

    int equivalenceClassesCount = 0;
    for (ui32 featureIdx = 0; featureIdx < featuresCombinations.size(); ++featureIdx) {
        int currentFeature = sortedBinFeatures[featureIdx];
        int previousFeature = featureIdx == 0 ? -1 : sortedBinFeatures[featureIdx - 1];
        if (featureIdx == 0 || featuresCombinations[currentFeature] != featuresCombinations[previousFeature]) {
            combinationClassFeatures->push_back(featuresCombinations[currentFeature]);
            ++equivalenceClassesCount;
        }
        for (size_t binBucketId = featureFirstBinBucket[currentFeature];
             binBucketId < featureFirstBinBucket[currentFeature] + featureBucketSizes[currentFeature];
             ++binBucketId)
        {
            (*binFeatureCombinationClass)[binBucketId] = equivalenceClassesCount - 1;
        }
    }
}

static double CalcAverageApprox(const TVector<double>& averageApproxByClass) {
    double result = 0;
    for (double value : averageApproxByClass) {
        result += value;
    }
    return result / averageApproxByClass.size();
}

bool IsPrepareTreesCalcShapValues(
    const TFullModel& model,
    const TDataProvider* dataset,
    EPreCalcShapValues mode
) {
    switch (mode) {
        case EPreCalcShapValues::UsePreCalc:
            CB_ENSURE(model.IsOblivious(), "UsePreCalc mode can be used only for symmetric trees.");
            return true;
        case EPreCalcShapValues::NoPreCalc:
            return false;
        case EPreCalcShapValues::Auto:
            if (dataset==nullptr) {
                return true;
            } else {
                if (!model.IsOblivious()) {
                    return false;
                }
                const size_t treeCount = model.GetTreeCount();
                const TModelTrees& forest = *model.ModelTrees;
                double treesAverageLeafCount = forest.GetLeafValues().size() / treeCount;
                return treesAverageLeafCount < dataset->ObjectsGrouping->GetObjectCount();
            }
    }
    Y_UNREACHABLE();
}

static bool AreApproxesZeroForLastClass(
    const TModelTrees& forest,
    size_t treeIdx) {

    const int approxDimension = forest.GetDimensionsCount();
    const double Eps = 1e-12;
    if (forest.IsOblivious()) {
        auto firstLeafPtr = forest.GetFirstLeafPtrForTree(treeIdx);
        const size_t maxDepth = forest.GetTreeSizes()[treeIdx];
        for (size_t leafIdx = 0; leafIdx < (size_t(1) << maxDepth); ++leafIdx) {
            if (fabs(firstLeafPtr[leafIdx * approxDimension + approxDimension - 1]) > Eps){
                return false;
            }
        }
    } else {
        const int totalNodesCount = forest.GetNonSymmetricNodeIdToLeafId().size();
        const bool isLastTree = treeIdx == forest.GetTreeStartOffsets().size() - 1;
        const size_t startOffset = forest.GetTreeStartOffsets()[treeIdx];
        const size_t endOffset = isLastTree ? totalNodesCount : forest.GetTreeStartOffsets()[treeIdx + 1];
        for (size_t nodeIdx = startOffset; nodeIdx < endOffset; ++nodeIdx) {
            size_t leafIdx = forest.GetNonSymmetricNodeIdToLeafId()[nodeIdx];
            if (leafIdx < forest.GetLeafValues().size() && fabs(forest.GetLeafValues()[leafIdx + approxDimension]) > Eps) {
                return false;
            }
        }
    }
    return true;
}

static bool IsMultiClass(const TFullModel& model) {
    return model.ModelTrees->GetDimensionsCount() > 1;
}

TMaybe<ELossFunction> TryGuessModelMultiClassLoss(const TFullModel& model) {
    TString lossFunctionName = model.GetLossFunctionName();
    if (lossFunctionName) {
        return FromString<ELossFunction>(lossFunctionName);
    } else {
        const auto& forest = *model.ModelTrees;
        bool approxesAreZeroForLastClass = true;
        for (size_t treeIdx = 0; treeIdx < model.GetTreeCount(); ++treeIdx) {
            approxesAreZeroForLastClass &= AreApproxesZeroForLastClass(forest, treeIdx);
        }
        return approxesAreZeroForLastClass ? TMaybe<ELossFunction>(ELossFunction::MultiClass) : Nothing();
    }
}

static void CalcTreeStats(
    const TModelTrees& forest,
    const TVector<double>& leafWeights,
    bool isMultiClass,
    TShapPreparedTrees* preparedTrees
) {
    const size_t treeCount = forest.GetTreeCount();
    for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        preparedTrees->SubtreeWeightsForAllTrees[treeIdx] = CalcSubtreeWeightsForTree(forest, leafWeights, treeIdx);
        preparedTrees->MeanValuesForAllTrees[treeIdx]
           = CalcMeanValueForTree(forest, preparedTrees->SubtreeWeightsForAllTrees[treeIdx], treeIdx);
        preparedTrees->AverageApproxByTree[treeIdx] = isMultiClass ? CalcAverageApprox(preparedTrees->MeanValuesForAllTrees[treeIdx]) : 0;
    }
}

static void InitPreparedTrees(
    const TFullModel& model,
    const TDataProvider* dataset, // can be nullptr if model has LeafWeights
    const TDataProviderPtr referenceDataset,
    EPreCalcShapValues mode,
    bool calcInternalValues,
    ECalcTypeShapValues calcType,
    EModelOutputType modelOutputType,
    NPar::TLocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees 
) {
    const size_t treeCount = model.GetTreeCount();
    // use only if model.ModelTrees->LeafWeights is empty
    TVector<double> leafWeights;
    if (model.ModelTrees->GetLeafWeights().empty()) {
        CB_ENSURE(
                dataset,
                "PrepareTrees requires either non-empty LeafWeights in model or provided dataset"
        );
        CB_ENSURE(dataset->ObjectsGrouping->GetObjectCount() != 0, "To calculate shap values, dataset must contain objects.");
        CB_ENSURE(dataset->MetaInfo.GetFeatureCount() > 0, "To calculate shap values, dataset must contain features.");
        leafWeights = CollectLeavesStatistics(*dataset, model, localExecutor);
    }

    preparedTrees->CalcShapValuesByLeafForAllTrees = IsPrepareTreesCalcShapValues(model, dataset, mode);

    if (!preparedTrees->CalcShapValuesByLeafForAllTrees) {
        TVector<double> modelLeafWeights(model.ModelTrees->GetLeafWeights().begin(), model.ModelTrees->GetLeafWeights().end());
        preparedTrees->LeafWeightsForAllTrees
            = modelLeafWeights.empty() ? leafWeights : modelLeafWeights;
    }

    preparedTrees->ShapValuesByLeafForAllTrees.resize(treeCount);
    preparedTrees->SubtreeWeightsForAllTrees.resize(treeCount);
    preparedTrees->MeanValuesForAllTrees.resize(treeCount);
    preparedTrees->AverageApproxByTree.resize(treeCount);
    preparedTrees->CalcInternalValues = calcInternalValues;

    const TModelTrees& forest = *model.ModelTrees;
    MapBinFeaturesToClasses(
        forest,
        &preparedTrees->BinFeatureCombinationClass,
        &preparedTrees->CombinationClassFeatures
    );
    if (calcType == ECalcTypeShapValues::IndependentTreeSHAP) {
        preparedTrees->IndependentTreeShapParams = TIndependentTreeShapParams(
            model,
            *dataset,
            *referenceDataset,
            preparedTrees->BinFeatureCombinationClass,
            modelOutputType,
            localExecutor
        );
    }
}

static void InitLeafWeights(
    const TFullModel& model,
    const TDataProvider* dataset,
    NPar::TLocalExecutor* localExecutor,
    TVector<double>* leafWeights
) {
    const auto& leafWeightsOfModels = model.ModelTrees->GetLeafWeights();
    if (leafWeightsOfModels.empty()) {
        CB_ENSURE(
                dataset,
                "To calculate shap values, either a model with leaf weights, or a dataset are required."
        );
        CB_ENSURE(dataset->ObjectsGrouping->GetObjectCount() != 0, "To calculate shap values, dataset must contain objects.");
        CB_ENSURE(dataset->MetaInfo.GetFeatureCount() > 0, "To calculate shap values, dataset must contain features.");
        *leafWeights = CollectLeavesStatistics(*dataset, model, localExecutor);
    } else {
        leafWeights->assign(leafWeightsOfModels.begin(), leafWeightsOfModels.end());
    }
}

static inline bool IsMultiClassification(const TFullModel& model) {
    ELossFunction modelLoss = ELossFunction::RMSE;
    if (IsMultiClass(model)) {
        TMaybe<ELossFunction> loss = TryGuessModelMultiClassLoss(model);
        if (loss) {
            modelLoss = *loss.Get();
        } else {
            CATBOOST_WARNING_LOG << "There is no loss_function parameter in the model, so it is considered as MultiClass" << Endl;
            modelLoss = ELossFunction::MultiClass;
        }
    }
    return (modelLoss == ELossFunction::MultiClass);
}

TShapPreparedTrees PrepareTrees(
    const TFullModel& model,
    const TDataProvider* dataset, // can be nullptr if model has LeafWeights
    const TDataProviderPtr referenceDataset,
    EPreCalcShapValues mode,
    ECalcTypeShapValues calcType,
    EModelOutputType modelOutputType,
    NPar::TLocalExecutor* localExecutor,
    bool calcInternalValues
) {
    TVector<double> leafWeights;
    InitLeafWeights(model, dataset, localExecutor, &leafWeights);
    TShapPreparedTrees preparedTrees;
    InitPreparedTrees(model, dataset, referenceDataset, mode, calcInternalValues, calcType, modelOutputType, localExecutor, &preparedTrees);
    const bool isMultiClass = IsMultiClassification(model);
    CalcTreeStats(*model.ModelTrees, leafWeights, isMultiClass, &preparedTrees);
    return preparedTrees;
}

TShapPreparedTrees PrepareTrees(
    const TFullModel& model,
    NPar::TLocalExecutor* localExecutor
) {
    CB_ENSURE(
        !model.ModelTrees->GetLeafWeights().empty(),
        "Model must have leaf weights or sample pool must be provided"
    );
    TShapPreparedTrees preparedTrees = PrepareTrees(model, nullptr, nullptr, EPreCalcShapValues::Auto, ECalcTypeShapValues::TreeSHAP, EModelOutputType::Raw, localExecutor);
    CalcShapValuesByLeaf(
        model,
        /*fixedFeatureParams*/ Nothing(),
        /*calcType*/ ECalcTypeShapValues::TreeSHAP,
        /*logPeriod*/ 0,
        preparedTrees.CalcInternalValues,
        localExecutor,
        &preparedTrees
    );
    return preparedTrees;
}