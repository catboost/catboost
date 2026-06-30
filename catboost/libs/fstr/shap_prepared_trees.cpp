#include "shap_prepared_trees.h"

#include "shap_values.h"
#include "util.h"

#include <catboost/private/libs/algo/features_data_helpers.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/data/features_layout_helpers.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/model/cpu/quantization.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/utility.h>
#include <util/generic/ymath.h>


using namespace NCB;


static TVector<double> CalcMeanValueForTree(
    const TModelTrees& forest,
    const TVector<TVector<double>>& subtreeWeights,
    TConstArrayRef<double> leafWeights,
    size_t treeIdx
) {
    const int approxDimension = forest.GetDimensionsCount();
    TVector<double> meanValue(approxDimension, 0.0);

    if (forest.IsOblivious()) {
        auto firstLeafPtr = forest.GetFirstLeafPtrForTree(treeIdx);
        const size_t maxDepth = forest.GetModelTreeData()->GetTreeSizes()[treeIdx];
        const auto& subtreeWeightsOnMaxDepth = subtreeWeights[maxDepth];
        for (size_t leafIdx = 0; leafIdx < (size_t(1) << maxDepth); ++leafIdx) {
            for (int dimension = 0; dimension < approxDimension; ++dimension) {
                meanValue[dimension] += firstLeafPtr[leafIdx * approxDimension + dimension]
                                     * subtreeWeightsOnMaxDepth[leafIdx];
            }
        }
    } else {
        const int totalNodesCount = forest.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId().size();
        const bool isLastTree = treeIdx == forest.GetModelTreeData()->GetTreeStartOffsets().size() - 1;
        const size_t startOffset = forest.GetModelTreeData()->GetTreeStartOffsets()[treeIdx];
        const size_t endOffset = isLastTree ? totalNodesCount : forest.GetModelTreeData()->GetTreeStartOffsets()[treeIdx + 1];
        const auto leafValues = forest.GetModelTreeData()->GetLeafValues();
        const auto nonSymmetricNodeIdToLeafId = forest.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId();
        const size_t leafValueCount = leafValues.size();
        for (size_t nodeIdx = startOffset; nodeIdx < endOffset; ++nodeIdx) {
            size_t leafIdx = nonSymmetricNodeIdToLeafId[nodeIdx];
            for (int dimension = 0; dimension < approxDimension; ++dimension) {
                if (leafIdx < leafValueCount) {
                    meanValue[dimension] += leafValues[leafIdx + dimension] * leafWeights[leafIdx / approxDimension];
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
    const int totalNodesCount = forest.GetModelTreeData()->GetTreeSplits().size();
    const bool isLastTree = static_cast<size_t>(treeIdx + 1) == forest.GetModelTreeData()->GetTreeStartOffsets().size();
    const int startOffset = forest.GetModelTreeData()->GetTreeStartOffsets()[treeIdx];
    const int endOffset = isLastTree ? totalNodesCount : forest.GetModelTreeData()->GetTreeStartOffsets()[treeIdx + 1];
    const int treeSize = endOffset - startOffset;
    const auto nonSymmetricStepNodes = forest.GetModelTreeData()->GetNonSymmetricStepNodes();

    TVector<size_t> reversedTree(treeSize, 0);
    for (int nodeIdx = startOffset; nodeIdx < endOffset; ++nodeIdx) {
        const int localIdx = nodeIdx - startOffset;
        const size_t leftDiff = nonSymmetricStepNodes[nodeIdx].LeftSubtreeDiff;
        const size_t rightDiff = nonSymmetricStepNodes[nodeIdx].RightSubtreeDiff;
        if (leftDiff != 0) {
            reversedTree[localIdx + leftDiff] = nodeIdx;
        }
        if (rightDiff != 0) {
            reversedTree[localIdx + rightDiff] = nodeIdx;
        }
    }
    return reversedTree;
}

static TVector<TVector<TVector<double>>> CalcSubtreeValuesForTree(
    const TModelTrees& forest,
    const TVector<TVector<double>>& subtreeWeights,
    const TVector<double>& leafWeights,
    size_t treeIdx
) {
    const size_t approxDimension = forest.GetDimensionsCount();
    TVector<TVector<TVector<double>>> subtreeValues;
    if (forest.IsOblivious()) {
        auto firstLeafPtr = forest.GetFirstLeafPtrForTree(treeIdx);
        const size_t treeDepth = forest.GetModelTreeData()->GetTreeSizes()[treeIdx];
        subtreeValues.resize(treeDepth + 1);
        size_t leafNum = size_t(1) << treeDepth;
        subtreeValues[treeDepth].resize(leafNum);
        for (size_t leafIdx = 0; leafIdx < leafNum; ++leafIdx) {
            subtreeValues[treeDepth][leafIdx].resize(approxDimension, 0.0);
            for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
                subtreeValues[treeDepth][leafIdx][dimension] = firstLeafPtr[leafIdx * approxDimension + dimension];
            }
        }
        for (int depth = treeDepth - 1; depth >= 0; --depth) {
            size_t subtreeNum = size_t(1) << depth;
            subtreeValues[depth].resize(subtreeNum);
            for (size_t subtreeIdx = 0; subtreeIdx < subtreeNum; ++subtreeIdx) {
                subtreeValues[depth][subtreeIdx].resize(approxDimension, 0.0);
                if (!FuzzyEquals(1 + subtreeWeights[depth][subtreeIdx], 1 + 0.0)) {
                    for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
                        subtreeValues[depth][subtreeIdx][dimension] =
                                subtreeValues[depth + 1][subtreeIdx * 2][dimension]
                                * subtreeWeights[depth + 1][subtreeIdx * 2] +
                                subtreeValues[depth + 1][subtreeIdx * 2 + 1][dimension]
                                * subtreeWeights[depth + 1][subtreeIdx * 2 + 1];
                        subtreeValues[depth][subtreeIdx][dimension] /= subtreeWeights[depth][subtreeIdx];
                    }
                }
            }
        }
    } else {
        const size_t startOffset = forest.GetModelTreeData()->GetTreeStartOffsets()[treeIdx];
        auto firstLeafPtr = &forest.GetModelTreeData()->GetLeafValues()[0];
        TVector<size_t> reversedTree = GetReversedSubtreeForNonObliviousTree(forest, treeIdx);
        subtreeValues.resize(1);
        subtreeValues[0].resize(reversedTree.size(), TVector<double>(approxDimension, 0.0));
        if (reversedTree.size() == 1) {
            size_t leafIdx = forest.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[startOffset];
            for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
                subtreeValues[0][0][dimension] = firstLeafPtr[leafIdx + dimension];
            }
        } else {
            for (size_t localIdx = reversedTree.size() - 1; localIdx > 0; --localIdx) {
                size_t leafIdx = forest.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[startOffset + localIdx];
                size_t leafWeightIdx = leafIdx / approxDimension;
                if (leafWeightIdx < leafWeights.size()) {
                    if (!FuzzyEquals(1 + leafWeights[leafWeightIdx], 1 + 0.0)) {
                        for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
                            subtreeValues[0][localIdx][dimension] +=
                                firstLeafPtr[leafIdx + dimension]
                                * leafWeights[leafWeightIdx];
                        }
                    }
                }
                if (!FuzzyEquals(1 + subtreeWeights[0][localIdx], 1 + 0.0)) {
                    for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
                        subtreeValues[0][reversedTree[localIdx] - startOffset][dimension] +=
                            subtreeValues[0][localIdx][dimension];
                    }
                }
            }
            for (int localIdx = reversedTree.size() - 1; localIdx >= 0; --localIdx) {
                if (!FuzzyEquals(1 + subtreeWeights[0][localIdx], 1 + 0.0)) {
                    for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
                        subtreeValues[0][localIdx][dimension] /= subtreeWeights[0][localIdx];
                    }
                }
            }
        }
    }

    return subtreeValues;
}

static TVector<TVector<double>> CalcSubtreeWeightsForTree(
    const TModelTrees& forest,
    const TVector<double>& leafWeights,
    int treeIdx
) {
    TVector<TVector<double>> subtreeWeights;
    if (forest.IsOblivious()) {
        const int treeDepth = forest.GetModelTreeData()->GetTreeSizes()[treeIdx];
        subtreeWeights.resize(treeDepth + 1);
        subtreeWeights[treeDepth].resize(size_t(1) << treeDepth);
        TArrayRef<double> subtreeWeightsOnTreeDepth = MakeArrayRef(subtreeWeights[treeDepth]);
        auto applyData = forest.GetApplyData();
        const int weightOffset = applyData->TreeFirstLeafOffsets[treeIdx] / forest.GetDimensionsCount();

        for (size_t nodeIdx = 0; nodeIdx < size_t(1) << treeDepth; ++nodeIdx) {
            subtreeWeightsOnTreeDepth[nodeIdx] = leafWeights[weightOffset + nodeIdx];
        }

        for (int depth = treeDepth - 1; depth >= 0; --depth) {
            const size_t nodeCount = size_t(1) << depth;
            subtreeWeights[depth].resize(nodeCount);
            TConstArrayRef<double> subtreeWeightsOnChild = MakeConstArrayRef(subtreeWeights[depth + 1]);
            TArrayRef<double> subtreeWeightsOnParent = MakeArrayRef(subtreeWeights[depth]);
            for (size_t nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx) {
                subtreeWeightsOnParent[nodeIdx] = subtreeWeightsOnChild[nodeIdx * 2] + subtreeWeightsOnChild[nodeIdx * 2 + 1];
            }
        }
    } else {
        const int startOffset = forest.GetModelTreeData()->GetTreeStartOffsets()[treeIdx];
        TVector<size_t> reversedTree = GetReversedSubtreeForNonObliviousTree(forest, treeIdx);
        subtreeWeights.resize(1); // with respect to NonSymmetric format of TObliviousTree
        subtreeWeights[0].resize(reversedTree.size(), 0);
        TArrayRef<double> subtreeWeightsOnParent = MakeArrayRef(subtreeWeights[0]);
        if (reversedTree.size() == 1) {
            subtreeWeightsOnParent[0] = leafWeights[forest.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[startOffset] / forest.GetDimensionsCount()];
        } else {
            for (size_t localIdx = reversedTree.size() - 1; localIdx > 0; --localIdx) {
                size_t leafIdx = forest.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[startOffset + localIdx] / forest.GetDimensionsCount();
                if (leafIdx < leafWeights.size()) {
                    subtreeWeightsOnParent[localIdx] += leafWeights[leafIdx];
                }
                subtreeWeightsOnParent[reversedTree[localIdx] - startOffset] += subtreeWeightsOnParent[localIdx];
            }
        }
    }
    return subtreeWeights;
}

static void MapBinFeaturesToClasses(
    const TFullModel& model,
    TVector<int>* binFeatureCombinationClass,
    TVector<TVector<int>>* combinationClassFeatures
) {
    const NCB::TFeaturesLayout layout = MakeFeaturesLayout(model);
    TVector<TVector<int>> featuresCombinations;
    TVector<size_t> featureBucketSizes;
    const TModelTrees& forest = *model.ModelTrees;

    for (const TFloatFeature& floatFeature : forest.GetFloatFeatures()) {
        if (!floatFeature.UsedInModel()) {
            continue;
        }
        featuresCombinations.emplace_back(1, floatFeature.Position.FlatIndex);
        featureBucketSizes.push_back(floatFeature.Borders.size());
    }

    for (const TOneHotFeature& oneHotFeature : forest.GetOneHotFeatures()) {
        featuresCombinations.emplace_back(
            1,
            (int)layout.GetExternalFeatureIdx(oneHotFeature.CatFeatureIndex,
            EFeatureType::Categorical)
        );
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

    for (const TEstimatedFeature& estimatedFeature: forest.GetEstimatedFeatures()) {
        featuresCombinations.emplace_back(
            1,
            (int)layout.GetExternalFeatureIdx(
                estimatedFeature.ModelEstimatedFeature.SourceFeatureId,
                EstimatedSourceFeatureTypeToFeatureType(estimatedFeature.ModelEstimatedFeature.SourceFeatureType)
            )
        );
        featureBucketSizes.push_back(estimatedFeature.Borders.size());
    }

    TVector<size_t> featureFirstBinBucket(featureBucketSizes.size(), 0);
    for (size_t idx : xrange((size_t)1, featureBucketSizes.size())) {
        featureFirstBinBucket[idx] = featureFirstBinBucket[idx - 1] + featureBucketSizes[idx - 1];
    }
    TVector<int> sortedBinFeatures(featuresCombinations.size());
    Iota(sortedBinFeatures.begin(), sortedBinFeatures.end(), 0);
    StableSort(
        sortedBinFeatures.begin(),
        sortedBinFeatures.end(),
        [featuresCombinations](int feature1, int feature2) {
            return featuresCombinations[feature1] < featuresCombinations[feature2];
        }
    );

    *binFeatureCombinationClass = TVector<int>(forest.GetBinaryFeaturesFullCount());
    *combinationClassFeatures = TVector<TVector<int>>();

    int equivalenceClassesCount = 0;
    for (ui32 featureIdx : xrange(featuresCombinations.size())) {
        int currentFeature = sortedBinFeatures[featureIdx];
        int previousFeature = featureIdx == 0 ? -1 : sortedBinFeatures[featureIdx - 1];
        if (featureIdx == 0 || featuresCombinations[currentFeature] != featuresCombinations[previousFeature]) {
            combinationClassFeatures->push_back(featuresCombinations[currentFeature]);
            ++equivalenceClassesCount;
        }
        for (size_t bucketIdx : xrange(featureBucketSizes[currentFeature])) {
            ui32 binBucketId = bucketIdx + featureFirstBinBucket[currentFeature];
            (*binFeatureCombinationClass)[binBucketId] = equivalenceClassesCount - 1;
        }
    }
}

static double CalcAverageApprox(const TVector<double>& averageApproxByClass) {
    double result = Accumulate(averageApproxByClass.begin(), averageApproxByClass.end(), 0.0);
    return result / averageApproxByClass.size();
}

bool IsPrepareTreesCalcShapValues(
    const TFullModel& model,
    TMaybe<ui32> datasetObjectCount,
    EPreCalcShapValues mode
) {
    switch (mode) {
        case EPreCalcShapValues::UsePreCalc:
            CB_ENSURE(model.IsOblivious(), "UsePreCalc mode can be used only for symmetric trees.");
            return true;
        case EPreCalcShapValues::NoPreCalc:
            return false;
        case EPreCalcShapValues::Auto: {
            if (!datasetObjectCount) {
                return true;
            }
            if (!model.IsOblivious()) {
                return false;
            }
            const size_t treeCount = model.GetTreeCount();
            const TModelTrees& forest = *model.ModelTrees;
            double treesAverageLeafCount = forest.GetModelTreeData()->GetLeafValues().size() / treeCount;
            return treesAverageLeafCount < *datasetObjectCount;
        }
        default:
            CB_ENSURE(false, "Unexpected type of SHAP precalculation");
    }
}

static bool AreApproxesZeroForLastClass(
    const TModelTrees& forest,
    size_t treeIdx
) {
    const int approxDimension = forest.GetDimensionsCount();
    const double Eps = 1e-12;
    if (forest.IsOblivious()) {
        auto firstLeafPtr = forest.GetFirstLeafPtrForTree(treeIdx);
        const size_t maxDepth = forest.GetModelTreeData()->GetTreeSizes()[treeIdx];
        for (size_t leafIdx = 0; leafIdx < (size_t(1) << maxDepth); ++leafIdx) {
            if (fabs(firstLeafPtr[leafIdx * approxDimension + approxDimension - 1]) > Eps){
                return false;
            }
        }
    } else {
        const int totalNodesCount = forest.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId().size();
        const bool isLastTree = treeIdx == forest.GetModelTreeData()->GetTreeStartOffsets().size() - 1;
        const size_t startOffset = forest.GetModelTreeData()->GetTreeStartOffsets()[treeIdx];
        const size_t endOffset = isLastTree ? totalNodesCount : forest.GetModelTreeData()->GetTreeStartOffsets()[treeIdx + 1];
        for (size_t nodeIdx = startOffset; nodeIdx < endOffset; ++nodeIdx) {
            size_t leafIdx = forest.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[nodeIdx];
            if (leafIdx < forest.GetModelTreeData()->GetLeafValues().size() && fabs(forest.GetModelTreeData()->GetLeafValues()[leafIdx + approxDimension]) > Eps) {
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
    ECalcTypeShapValues calcType,
    TShapPreparedTrees* preparedTrees
) {
    const size_t treeCount = forest.GetTreeCount();
    for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        preparedTrees->SubtreeWeightsForAllTrees[treeIdx] = CalcSubtreeWeightsForTree(forest, leafWeights, treeIdx);
        preparedTrees->MeanValuesForAllTrees[treeIdx] = CalcMeanValueForTree(
            forest,
            preparedTrees->SubtreeWeightsForAllTrees[treeIdx],
            preparedTrees->LeafWeightsForAllTrees,
            treeIdx
        );
        if (calcType == ECalcTypeShapValues::Approximate) {
            preparedTrees->SubtreeValuesForAllTrees[treeIdx] =
                    CalcSubtreeValuesForTree(forest, preparedTrees->SubtreeWeightsForAllTrees[treeIdx],
                                             leafWeights, treeIdx);
        }
        preparedTrees->AverageApproxByTree[treeIdx] = isMultiClass ? CalcAverageApprox(preparedTrees->MeanValuesForAllTrees[treeIdx]) : 0;
    }
}

static void InitPreparedTreesWithoutIndependent(
    const TFullModel& model,
    TMaybe<ui32> datasetObjectCount,
    EPreCalcShapValues mode,
    bool calcInternalValues,
    ECalcTypeShapValues calcType,
    TShapPreparedTrees* preparedTrees
) {
    const size_t treeCount = model.GetTreeCount();
    preparedTrees->CalcShapValuesByLeafForAllTrees = IsPrepareTreesCalcShapValues(
        model,
        datasetObjectCount,
        mode
    );

    preparedTrees->ShapValuesByLeafForAllTrees.resize(treeCount);
    preparedTrees->SubtreeWeightsForAllTrees.resize(treeCount);
    preparedTrees->MeanValuesForAllTrees.resize(treeCount);
    if (calcType == ECalcTypeShapValues::Approximate) {
        preparedTrees->SubtreeValuesForAllTrees.resize(treeCount);
    }
    preparedTrees->AverageApproxByTree.resize(treeCount);
    preparedTrees->CalcInternalValues = calcInternalValues;

    MapBinFeaturesToClasses(
        model,
        &preparedTrees->BinFeatureCombinationClass,
        &preparedTrees->CombinationClassFeatures
    );
}

static void InitLeafWeights(
    const TFullModel& model,
    bool needSumModelAndDatasetWeights,
    TConstArrayRef<double> leafWeightsFromDataset, // can be empty
    TVector<double>* leafWeights
) {
    const auto leafWeightsOfModels = model.ModelTrees->GetModelTreeData()->GetLeafWeights();

    if (leafWeightsOfModels.empty() || needSumModelAndDatasetWeights) {
        CB_ENSURE_INTERNAL(
            !leafWeightsFromDataset.empty(),
            "Leaf weights from dataset are not provided"
        );
        leafWeights->assign(leafWeightsFromDataset.begin(), leafWeightsFromDataset.end());
    }
    if (!leafWeightsOfModels.empty()) {
        if (!leafWeights->empty()) {
            auto dst = leafWeights->data();
            for (auto i : xrange(leafWeights->size())) {
                dst[i] += leafWeightsOfModels[i];
            }
        } else {
            leafWeights->assign(leafWeightsOfModels.begin(), leafWeightsOfModels.end());
        }
    }
}

static void InitLeafWeights(
    const TFullModel& model,
    bool fstrOnTrainPool,
    const TDataProvider* dataset,
    NPar::ILocalExecutor* localExecutor,
    TVector<double>* leafWeights
) {
    const auto leafWeightsOfModels = model.ModelTrees->GetModelTreeData()->GetLeafWeights();
    const bool needSumModelAndDatasetWeights = !fstrOnTrainPool && HasNonZeroApproxForZeroWeightLeaf(model);

    TVector<double> leafWeightsFromDataset;
    if (leafWeightsOfModels.empty() || needSumModelAndDatasetWeights) {
        CB_ENSURE(
            dataset,
            "To calculate shap values, either a model with leaf weights, or a dataset are required."
        );
        CB_ENSURE(dataset->ObjectsGrouping->GetObjectCount() != 0, "To calculate shap values, dataset must contain objects.");
        CB_ENSURE(dataset->MetaInfo.GetFeatureCount() > 0, "To calculate shap values, dataset must contain features.");
        leafWeightsFromDataset = CollectLeavesStatistics(*dataset, model, localExecutor);
    }
    InitLeafWeights(model, needSumModelAndDatasetWeights, leafWeightsFromDataset, leafWeights);
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


TShapPreparedTrees PrepareTreesWithoutIndependent(
    const TFullModel& model,
    i64 datasetObjectCount,
    bool needSumModelAndDatasetWeights,
    TConstArrayRef<double> leafWeightsFromDataset,
    EPreCalcShapValues mode,
    bool calcInternalValues,
    ECalcTypeShapValues calcType
) {
    TShapPreparedTrees preparedTrees;
    InitLeafWeights(model, needSumModelAndDatasetWeights, leafWeightsFromDataset, &preparedTrees.LeafWeightsForAllTrees);
    InitPreparedTreesWithoutIndependent(
        model,
        (datasetObjectCount > 0) ? TMaybe<ui32>(SafeIntegerCast<ui32>(datasetObjectCount)) : Nothing(),
        mode,
        calcInternalValues,
        calcType,
        &preparedTrees
    );
    const bool isMultiClass = IsMultiClassification(model);
    CalcTreeStats(*model.ModelTrees, preparedTrees.LeafWeightsForAllTrees, isMultiClass, calcType, &preparedTrees);
    return preparedTrees;
}


TShapPreparedTrees PrepareTrees(
    const TFullModel& model,
    const TDataProvider* dataset, // can be nullptr if model has LeafWeights
    const TDataProviderPtr referenceDataset,
    EPreCalcShapValues mode,
    NPar::ILocalExecutor* localExecutor,
    bool calcInternalValues,
    ECalcTypeShapValues calcType,
    EExplainableModelOutput modelOutputType,
    bool fstrOnTrainPool
) {
    TShapPreparedTrees preparedTrees;
    InitLeafWeights(model, fstrOnTrainPool, dataset, localExecutor, &preparedTrees.LeafWeightsForAllTrees);
    InitPreparedTreesWithoutIndependent(
        model,
        dataset ? TMaybe<ui32>(dataset->GetObjectCount()) : Nothing(),
        mode,
        calcInternalValues,
        calcType,
        &preparedTrees
    );
    if (calcType == ECalcTypeShapValues::Independent) {
        preparedTrees.IndependentTreeShapParams = TIndependentTreeShapParams(
            model,
            *dataset,
            *referenceDataset,
            modelOutputType,
            localExecutor
        );
    }
    const bool isMultiClass = IsMultiClassification(model);
    CalcTreeStats(*model.ModelTrees, preparedTrees.LeafWeightsForAllTrees, isMultiClass, calcType, &preparedTrees);
    return preparedTrees;
}

TShapPreparedTrees PrepareTrees(
    const TFullModel& model,
    NPar::ILocalExecutor* localExecutor
) {
    CB_ENSURE(
        !model.ModelTrees->GetModelTreeData()->GetLeafWeights().empty(),
        "Model must have leaf weights or sample pool must be provided"
    );
    TShapPreparedTrees preparedTrees = PrepareTrees(model, nullptr, nullptr, EPreCalcShapValues::Auto, localExecutor);
    CalcShapValuesByLeaf(
        model,
        /*fixedFeatureParams*/ Nothing(),
        /*logPeriod*/ 0,
        preparedTrees.CalcInternalValues,
        localExecutor,
        &preparedTrees
    );
    return preparedTrees;
}

