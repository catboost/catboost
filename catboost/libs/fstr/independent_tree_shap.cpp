#include "independent_tree_shap.h"
#include "shap_prepared_trees.h"

#include "util.h"


#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/model/cpu/quantization.h>
#include <catboost/libs/model/eval_processing.h>
#include <catboost/private/libs/algo/apply.h>
#include <catboost/private/libs/algo/features_data_helpers.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/private/libs/target/data_providers.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/utility.h>
#include <util/generic/ymath.h>

using namespace NCB;

static TContribution SumContributions(
    const TContribution& lhs,
    const TContribution& rhs
) {
    CB_ENSURE_INTERNAL(
        lhs.PositiveContribution.size() == rhs.PositiveContribution.size(), 
        "Contributions have different sizes");
    TContribution result{lhs.PositiveContribution.size()};
    const auto approxDimension = result.PositiveContribution.size();
    for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
        result.PositiveContribution[dimension] = lhs.PositiveContribution[dimension] + rhs.PositiveContribution[dimension];
        result.NegativeContribution[dimension] = lhs.NegativeContribution[dimension] + rhs.NegativeContribution[dimension];
    }
    return result;
}

// Independent Tree SHAP
TContribution TInternalIndependentTreeShapCalcer::Calc(
    int depth,
    size_t nodeIdx,
    ui32 uniqueFeaturesCount,
    ui32 featureMatchedDatasetSampleCount
) {
    if (depth == DepthOfTree) {
        TContribution contribution{ApproxDimension}; // (pos, neg) = (0, 0)
        if (featureMatchedDatasetSampleCount == 0) {
            for (size_t dimension = 0; dimension < ApproxDimension; ++dimension) {
                double value = LeafValuesPtr[nodeIdx * ApproxDimension + dimension];
                ShapValuesInternalByDepth[depth][dimension] += value;
            }
        }
        if (uniqueFeaturesCount != 0) {
            if (featureMatchedDatasetSampleCount != 0) {
                double weight = Weights[featureMatchedDatasetSampleCount - 1][uniqueFeaturesCount];
                for (size_t dimension = 0; dimension < ApproxDimension; ++dimension) {
                    contribution.PositiveContribution[dimension] = weight * LeafValuesPtr[nodeIdx * ApproxDimension + dimension]; 
                }
            }
            if (featureMatchedDatasetSampleCount != uniqueFeaturesCount) {
                double weight = -Weights[featureMatchedDatasetSampleCount][uniqueFeaturesCount];
                for (size_t dimension = 0; dimension < ApproxDimension; ++dimension) {
                    contribution.NegativeContribution[dimension] = weight * LeafValuesPtr[nodeIdx * ApproxDimension + dimension];
                }
            }
        }
        return contribution;
    }
    const int combinationClass = BinFeatureCombinationClassByDepth[depth];
    constexpr size_t None = (size_t)(-1);
    size_t nextNodeIdx = None;
	const size_t remainingDepth = DepthOfTree - depth - 1;
    // for current object, who fall down to leadIdx [documentLeafIdx]
    const bool isGoRight = (DocumentLeafIdx >> remainingDepth) & 1;
    const size_t goNodeIdx = nodeIdx * 2 + isGoRight;
    // for current reference object, who fall down to leadIdx [documentLeafIdx]
    const bool isGoRightReference = (DocumentLeafIdxReference >> remainingDepth) & 1;
    const size_t goNodeIdxReference = nodeIdx * 2 + isGoRightReference;             
    if (ListOfFeaturesDocumentLeaf[combinationClass] > 0) {
        nextNodeIdx = goNodeIdx;
    } else {
        if (ListOfFeaturesDocumentLeafReference[combinationClass] > 0) {
            nextNodeIdx = goNodeIdxReference;
        } else {
            if (goNodeIdx == goNodeIdxReference) {
                nextNodeIdx = goNodeIdx;
            } 
        }
    }
    if (nextNodeIdx != None) {
        return Calc(
            depth + 1,
            nextNodeIdx,
            uniqueFeaturesCount,
            featureMatchedDatasetSampleCount
        );
    }
    TContribution contribution{ApproxDimension};
    TContribution contributionReference{ApproxDimension};
    if (goNodeIdx != goNodeIdxReference) {
        // go to direction of sample
        ListOfFeaturesDocumentLeaf[combinationClass]++;
        contribution = Calc(
            depth + 1,
            goNodeIdx,
            uniqueFeaturesCount + 1,
            featureMatchedDatasetSampleCount + 1
        );
        ListOfFeaturesDocumentLeaf[combinationClass]--;
        // go to direction of reference sample
        ListOfFeaturesDocumentLeafReference[combinationClass]++;
        contributionReference = Calc(
            depth + 1,
            goNodeIdxReference,
            uniqueFeaturesCount + 1,
            featureMatchedDatasetSampleCount
        );
        ListOfFeaturesDocumentLeafReference[combinationClass]--;
    }
    for (size_t dimension = 0; dimension < ApproxDimension; ++dimension) {
        double value = contribution.PositiveContribution[dimension] + contributionReference.NegativeContribution[dimension];
        ShapValuesInternalByDepth[depth][dimension] += value;
    }
    return SumContributions(contribution, contributionReference);
}


static void SetValuesToShapValuesByReference(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<double>>& shapValueByDepthForLeaf,
    size_t treeIdx,
    TVector<TVector<double>>* shapValuesByReference,
    TVector<double>* meanValueByReference
) {
    const int approxDimension = forest.GetDimensionsCount();
    const size_t depthOfTree = forest.GetTreeSizes()[treeIdx];
    for (size_t depth = 0; depth < depthOfTree; ++depth) {
        const size_t remainingDepth = depthOfTree - depth - 1;
        const int combinationClass = binFeatureCombinationClass[
            forest.GetTreeSplits()[forest.GetTreeStartOffsets()[treeIdx] + remainingDepth]
        ];
        for (int dimension = 0; dimension < approxDimension; ++dimension) {
            (*shapValuesByReference)[dimension][combinationClass] +=
                shapValueByDepthForLeaf[depth][dimension];
        }
    }
    for (int dimension = 0; dimension < approxDimension; ++dimension) {
        (*meanValueByReference)[dimension] += shapValueByDepthForLeaf[depthOfTree][dimension];
    }
}

static void SetValuesToShapValuesByAllReferences(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<TVector<double>>>& shapValueByDepthForLeaf,
    const TVector<TVector<ui32>>& referenceIndices,
    const TVector<NCB::NModelEvaluation::TCalcerIndexType>& referenceLeafIndices,
    size_t treeIdx,
    bool isCalcForAllLeaves,
    TVector<TVector<TVector<double>>>* shapValuesForAllReferences,
    TVector<TVector<double>>* meanValueForAllReferences
) {
    if (isCalcForAllLeaves) {
        const size_t leafCount = (size_t(1) << forest.GetTreeSizes()[treeIdx]);
        for (size_t leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
            const auto& references = referenceIndices[leafIdx];
            for (size_t idx = 0; idx < references.size(); ++idx) {
                const size_t referenceIdx = references[idx];
                SetValuesToShapValuesByReference(
                    forest,
                    binFeatureCombinationClass,
                    shapValueByDepthForLeaf[leafIdx],
                    treeIdx,
                    &shapValuesForAllReferences->at(referenceIdx),
                    &meanValueForAllReferences->at(referenceIdx)
                );
            }
        }
    } else {
        for (size_t referenceIdx = 0; referenceIdx < referenceLeafIndices.size(); ++referenceIdx) {
            SetValuesToShapValuesByReference(
                forest,
                binFeatureCombinationClass,
                shapValueByDepthForLeaf[referenceLeafIndices[referenceIdx]],
                treeIdx,
                &shapValuesForAllReferences->at(referenceIdx),
                &meanValueForAllReferences->at(referenceIdx)
            );
        }
    }
}



static inline ui64 GetBinominalCoeffient(ui64 n, ui64 k) { 
    ui64 binomialCoefficient = 1; 
    if (k > n - k) {
        k = n - k; 
    }
    for (ui64 i = 0; i < k; ++i) { 
        binomialCoefficient *= (n - i);
        binomialCoefficient /= (i + 1); 
    } 
    return binomialCoefficient; 
}

static void UnpackInternalShapsForDocumentOneDimension(
    const TVector<double>& shapValuesInternal,
    const TVector<TVector<int>>& combinationClassFeatures,
    TVector<double>* shapValues
) {
    if (shapValuesInternal.empty()) {
        return;
    }
    for (int classIdx : xrange(combinationClassFeatures.ysize())) {
        const TVector<int> &flatFeatures = combinationClassFeatures[classIdx];
        double coefficient = flatFeatures.size();
        double addValue = shapValuesInternal[classIdx] / coefficient;
        for (int flatFeatureIdx : flatFeatures) {
            (*shapValues)[flatFeatureIdx] += addValue;
        }
    }
}

static void CalcObliviousShapValuesByDepthForLeaf(
    const TModelTrees& forest,
    const TVector<NCB::NModelEvaluation::TCalcerIndexType>& referenceLeafIndices,
    const TVector<int>& binFeatureCombinationClassByDepth,
    const TVector<TVector<double>>& weights,
    size_t classCount,
    size_t documentLeafIdx,
    size_t treeIdx,
    bool isCalcForAllLeafes,
    TVector<TVector<TVector<double>>>* shapValueByDepthBetweenLeaves
) {
    const size_t depthOfTree = forest.GetTreeSizes()[treeIdx];
    const size_t approxDimension = forest.GetDimensionsCount();
    const size_t leafCountInTree = (size_t(1) << forest.GetTreeSizes()[treeIdx]);
    const size_t leafCount = isCalcForAllLeafes ? leafCountInTree : referenceLeafIndices.size(); 
    for (size_t idx = 0; idx < leafCount; ++idx) {
        const size_t leafIdx = isCalcForAllLeafes ? idx : referenceLeafIndices[idx];
        TVector<TVector<double>>& shapValuesInternalByDepth = shapValueByDepthBetweenLeaves->at(leafIdx);
        shapValuesInternalByDepth.assign(depthOfTree + 1, TVector<double>(approxDimension, 0.0));
        TInternalIndependentTreeShapCalcer calcerIntenalShaps{
            forest,
            binFeatureCombinationClassByDepth,
            weights,
            classCount,
            documentLeafIdx,
            /*documentLeafIdxReference*/ leafIdx,
            treeIdx,
            &shapValuesInternalByDepth
        };
        calcerIntenalShaps.Calc();
    }
}



static inline double GetTransformData(
    const TTransformFunc& TransformFunction,
    EModelOutputType modelOutputType,
    double target,
    double approx
) {
    switch (modelOutputType) {
        case EModelOutputType::Probability: return 1.0 / (1 + exp(-approx));
        case EModelOutputType::LossFunction: return TransformFunction(target, approx);
        default: CB_ENSURE_INTERNAL(false, "Not recognized type of model output");
    }
}

void IndependentTreeShap(
    const TFullModel& model,
    const TShapPreparedTrees& preparedTrees,
    int flatFeatureCount,
    TConstArrayRef<NModelEvaluation::TCalcerIndexType> docIndexes,
    size_t documentIdx,
    TVector<TVector<double>>* shapValues
) {
    CB_ENSURE(model.IsOblivious(), "Calculation shap values on mode independent for non oblivious tree unimplemented");
    const auto& forest = *model.ModelTrees.Get();
    const size_t classCount = preparedTrees.CombinationClassFeatures.size();
    const int approxDimension = model.GetDimensionsCount();
    const auto& independentTreeShapParams = preparedTrees.IndependentTreeShapParams.GetRef();
    const size_t referenceCount = independentTreeShapParams.ReferenceLeafIndicesForAllTrees[0].size();
    TVector<TVector<TVector<double>>> shapValuesForAllReferences(referenceCount);
    for (size_t referenceIdx = 0; referenceIdx < referenceCount; ++referenceIdx) {
        shapValuesForAllReferences[referenceIdx].assign(approxDimension, TVector<double>(classCount, 0.0));
    }
    TVector<TVector<double>> meanValueForAllReferences;
    meanValueForAllReferences.assign(referenceCount, TVector<double>(approxDimension, 0.0));
    const size_t treeCount = model.GetTreeCount();
    for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        if (preparedTrees.CalcShapValuesByLeafForAllTrees) {
            SetValuesToShapValuesByAllReferences(
                forest,
                preparedTrees.BinFeatureCombinationClass,
                independentTreeShapParams.ShapValueByDepthBetweenLeavesForAllTrees[treeIdx][docIndexes[treeIdx]],
                independentTreeShapParams.ReferenceIndicesForAllTrees[treeIdx],
                independentTreeShapParams.ReferenceLeafIndicesForAllTrees[treeIdx],
                treeIdx,
                /*isCalcForAllLeaves*/ false,
                &shapValuesForAllReferences,
                &meanValueForAllReferences
            );
        } else {
            const size_t leafCount = (size_t(1) << forest.GetTreeSizes()[treeIdx]);
            TVector<TVector<TVector<double>>> shapValueByDepthBetweenLeaves(leafCount);
            CalcObliviousShapValuesByDepthForLeaf(
                forest,
                independentTreeShapParams.ReferenceLeafIndicesForAllTrees[treeIdx],
                independentTreeShapParams.BinFeatureCombinationClassByDepthForAllTrees[treeIdx],
                independentTreeShapParams.Weights,
                preparedTrees.CombinationClassFeatures.size(),
                docIndexes[treeIdx],
                treeIdx,
                independentTreeShapParams.IsCalcForAllLeafesForAllTrees[treeIdx],
                &shapValueByDepthBetweenLeaves
            );
            SetValuesToShapValuesByAllReferences(
                forest,
                preparedTrees.BinFeatureCombinationClass,
                shapValueByDepthBetweenLeaves,
                independentTreeShapParams.ReferenceIndicesForAllTrees[treeIdx],
                independentTreeShapParams.ReferenceLeafIndicesForAllTrees[treeIdx],
                treeIdx,
                independentTreeShapParams.IsCalcForAllLeafesForAllTrees[treeIdx],
                &shapValuesForAllReferences,
                &meanValueForAllReferences
            );
        }
    }

    shapValues->assign(approxDimension, TVector<double>(flatFeatureCount + 1, 0.0));
    EModelOutputType modelOutputType = independentTreeShapParams.ModelOutputType;
    const bool isNotRawOutputType = (EModelOutputType::Raw != modelOutputType);
    const auto& transformFunc = independentTreeShapParams.TransformFunction;
    const auto& approxOfDataset = independentTreeShapParams.ApproxOfDataset;
    const auto& approxOfReferenceDataset = independentTreeShapParams.ApproxOfReferenceDataset;
    TConstArrayRef<double> targetOfDocumentRef = MakeConstArrayRef(independentTreeShapParams.TargetOfDataset[documentIdx]);
    TConstArrayRef<double> transformedTargetOfDocumentRef = MakeConstArrayRef(independentTreeShapParams.TransformedTargetOfDataset[documentIdx]);

    for (int dimension = 0; dimension < approxDimension; ++dimension) {
        TConstArrayRef<double> approxOfReferenceDatasetRef = MakeConstArrayRef(approxOfReferenceDataset[dimension]); 
        TArrayRef<double> shapValuesRef = MakeArrayRef((*shapValues)[dimension]);
        const double approxOfDocument = approxOfDataset[dimension][documentIdx];
        const double targetOfDocument = targetOfDocumentRef[dimension];
        const double transformedTargetOfDocument = transformedTargetOfDocumentRef[dimension];

        for (size_t referenceIdx = 0; referenceIdx < referenceCount; ++referenceIdx) {
            TVector<double> shapValueOneDimension(flatFeatureCount);
            UnpackInternalShapsForDocumentOneDimension(
                shapValuesForAllReferences[referenceIdx][dimension],
                preparedTrees.CombinationClassFeatures,
                &shapValueOneDimension
            );
            double rescaleCoefficient = referenceCount;
            double transformedCoeffient = 1.0;
            if (isNotRawOutputType && approxOfDocument != approxOfReferenceDatasetRef[referenceIdx]) {
                double transformReferenceTarget = GetTransformData(
                    transformFunc,
                    modelOutputType,
                    targetOfDocument,
                    approxOfReferenceDatasetRef[referenceIdx]
                );
                transformedCoeffient = (transformedTargetOfDocument - transformReferenceTarget) /
                    (approxOfDocument - approxOfReferenceDatasetRef[referenceIdx]);
            }
            double totalCoeffient = transformedCoeffient / rescaleCoefficient;
            for (int flatFeatureIdx = 0; flatFeatureIdx < flatFeatureCount; ++flatFeatureIdx) {
                shapValuesRef[flatFeatureIdx] +=
                    shapValueOneDimension[flatFeatureIdx] * totalCoeffient;
            }
            if (isNotRawOutputType) {
                shapValuesRef[flatFeatureCount] += GetTransformData(
                    transformFunc,
                    modelOutputType,
                    0,
                    meanValueForAllReferences[referenceIdx][dimension]
                ) / rescaleCoefficient;
            } else {
                shapValuesRef[flatFeatureCount] += (meanValueForAllReferences[referenceIdx][dimension] / rescaleCoefficient);
            }
        }
    }
    if (approxDimension == 1) {
        (*shapValues)[0][flatFeatureCount] += model.GetScaleAndBias().Bias;
    }
}


static TVector<TVector<double>> CalcWeightsForIndependentTreeShap(const TFullModel& model) {
    const TModelTrees& forest = *model.ModelTrees;
    const auto treeSizes = forest.GetTreeSizes();
    const size_t maxTreeDepth = *MaxElement(treeSizes.begin(), treeSizes.end());
    TVector<TVector<double>> weights;
    weights.assign(maxTreeDepth + 1, TVector<double>(maxTreeDepth + 1, 0.0));
    for (ui64 n = 0; n <= maxTreeDepth; ++n) {
        for (ui64 k = 1; k <= maxTreeDepth; ++k) {
            double coefficient = k * GetBinominalCoeffient(k - 1, n);
            double value = 1.0 / coefficient;
            weights[n][k] = value;
        }
    }
    return weights;
}

double MSETransform(double target, double approx) {
    return (target - approx) * (target - approx); 
}

double LoglossTransform(double target, double approx) {
    return log(1 + exp(approx)) - target * approx;
}

static inline TTransformFunc GetTransformFunction(const NCatboostOptions::TLossDescription& metricDescription) {
    ELossFunction lossFunction = metricDescription.GetLossFunction();
    switch (lossFunction) {
        case ELossFunction::RMSE: return MSETransform;
        case ELossFunction::Logloss: return LoglossTransform;
        default: {
            CB_ENSURE(false, "Only RMSE and Logloss metric are explainable by shap Values at the moment");
        }
    }
    Y_UNREACHABLE();
}

TVector<TVector<double>> TransformVector(const TVector<TVector<double>>& vector) {    
    TVector<TVector<double>> transformedVector;
    if (vector.size() == 0) {
        return transformedVector;
    }

    transformedVector.assign(vector[0].size(), TVector<double>(vector.size(), 0.0));
    for (auto i : xrange(vector[0].size())) {
        for (auto j : xrange(vector.size())) {
            transformedVector[i][j] = vector[j][i];
        }
    }
    return transformedVector;
}

TVector<TVector<double>> GetTransformedTarget(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& targetData,
    const TTransformFunc& transformFunction
) {
    CB_ENSURE_INTERNAL(
        approx[0].size() == targetData.size() && approx.size() == targetData[0].size(),
        "Approx and target must have same sizes"
    );
    TVector<TVector<double>> transformedTarget(targetData.size());
    for (auto documentIdx : xrange(targetData.size())) {
        transformedTarget[documentIdx].resize(targetData[documentIdx].size());
        for (auto dimension : xrange(targetData[documentIdx].size())) {
            transformedTarget[documentIdx][dimension] = transformFunction(
                targetData[documentIdx][dimension],
                approx[dimension][documentIdx]
            );
        }
    }
    return transformedTarget;
}

void TIndependentTreeShapParams::InitTransformedData(
    const TFullModel& model,
    const TDataProvider& dataset,
    const NCatboostOptions::TLossDescription& metricDescription,
    NPar::TLocalExecutor* localExecutor
) {
    auto targetDataProvider =
        CreateModelCompatibleProcessedDataProvider(dataset, {metricDescription}, model, GetMonopolisticFreeCpuRam(), nullptr, localExecutor).TargetData;
    CB_ENSURE(targetDataProvider->GetTarget(), "Label must be provided");
    auto targetDataRef = *(targetDataProvider->GetTarget());
    TVector<TVector<double>> targetDataVector(targetDataRef.size());
    for (auto idx : xrange(targetDataRef.size())) {
        targetDataVector[idx].resize(targetDataRef[idx].size());
        for (auto idx1 : xrange(targetDataRef[idx].size())) {
            targetDataVector[idx][idx1] = targetDataRef[idx][idx1];
        }
        //targetDataVector[idx] = TVector<double>(targetDataRef[idx].begin(), targetDataRef[idx].end());
    }
    TargetOfDataset = TransformVector(targetDataVector);
    const size_t approxDimension = model.GetDimensionsCount();
    const size_t documentCount = dataset.ObjectsGrouping->GetObjectCount();
    TransformedTargetOfDataset.assign(documentCount, TVector<double>(approxDimension, 0.0));
    switch (ModelOutputType)
    {
        case EModelOutputType::Probability: {
            TransformedTargetOfDataset = TransformVector( 
                ApplyModelMulti(model, *dataset.ObjectsData, EPredictionType::Probability, 0, 0, localExecutor)
            );
            break;
        }
        case EModelOutputType::LossFunction : {
            TransformFunction = GetTransformFunction(metricDescription);
            TransformedTargetOfDataset = GetTransformedTarget(
                ApproxOfDataset,
                TargetOfDataset,
                TransformFunction
            );
            break;
        }
        default:
            break;
    }
 }

TIndependentTreeShapParams::TIndependentTreeShapParams(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TDataProvider& referenceDataset,
    const TVector<int>& binFeatureCombinationClass,
    EModelOutputType modelOutputType,
    NPar::TLocalExecutor* localExecutor
) {
    ModelOutputType = modelOutputType;
    Weights = CalcWeightsForIndependentTreeShap(model);
    NCatboostOptions::TLossDescription metricDescription;
    CB_ENSURE(TryGetObjectiveMetric(model, metricDescription), "Cannot calculate Shap values without metric, need model with params");
    CATBOOST_INFO_LOG << "Used " << metricDescription << " metric for fstr calculation" << Endl;
    // prepare data for reference dataset
    auto binarizedFeatures = MakeQuantizedFeaturesForEvaluator(model, *referenceDataset.ObjectsData);
    TVector<NModelEvaluation::TCalcerIndexType> indexes(binarizedFeatures->GetObjectsCount() * model.GetTreeCount());
    model.GetCurrentEvaluator()->CalcLeafIndexes(binarizedFeatures.Get(), 0, model.GetTreeCount(), indexes);
    const size_t referenceCount = binarizedFeatures->GetObjectsCount();
    const size_t treeCount = model.GetTreeCount();
    const TModelTrees& forest = *model.ModelTrees;
    ReferenceLeafIndicesForAllTrees.assign(treeCount, TVector<NCB::NModelEvaluation::TCalcerIndexType>(referenceCount, 0));
    ReferenceIndicesForAllTrees.resize(treeCount);
    for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        const size_t leafCount = size_t(1) << forest.GetTreeSizes()[treeIdx];
        ReferenceIndicesForAllTrees[treeIdx].resize(leafCount);
        const bool isCalcForAllLeafes = (referenceCount >= leafCount);
        IsCalcForAllLeafesForAllTrees.emplace_back(isCalcForAllLeafes);
    }
    for (size_t referenceIdx = 0; referenceIdx < referenceCount; ++referenceIdx) {
        const auto indexesForTrees = MakeArrayRef(indexes.data() + referenceIdx * model.GetTreeCount(), model.GetTreeCount());
        for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
            const auto leafIdx = indexesForTrees[treeIdx];
            ReferenceLeafIndicesForAllTrees[treeIdx][referenceIdx] = leafIdx;
            ReferenceIndicesForAllTrees[treeIdx][leafIdx].emplace_back(referenceIdx);
        }
    }

    BinFeatureCombinationClassByDepthForAllTrees.resize(treeCount);
    for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        const size_t depthOfTree = forest.GetTreeSizes()[treeIdx];
        BinFeatureCombinationClassByDepthForAllTrees[treeIdx].reserve(depthOfTree);
        for (size_t depth = 0; depth < depthOfTree; ++depth) {
			const size_t remainingDepth = depthOfTree - depth - 1;
            const int combinationClass = binFeatureCombinationClass[
                forest.GetTreeSplits()[forest.GetTreeStartOffsets()[treeIdx] + remainingDepth]
            ];
            BinFeatureCombinationClassByDepthForAllTrees[treeIdx].emplace_back(combinationClass);
        }
    }
    ApproxOfDataset = ApplyModelMulti(model, *dataset.ObjectsData, EPredictionType::RawFormulaVal, 0, 0, localExecutor);
    ApproxOfReferenceDataset = ApplyModelMulti(model, *referenceDataset.ObjectsData, EPredictionType::RawFormulaVal, 0, 0, localExecutor);
    InitTransformedData(model, dataset, metricDescription, localExecutor);
    ShapValueByDepthBetweenLeavesForAllTrees.resize(treeCount);
}

void CalcIndependentTreeShapValuesByLeafForTreeBlock(
    const TModelTrees& forest,
    size_t treeIdx,
    TShapPreparedTrees* preparedTrees
) {
    for (size_t leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
        ////
        const size_t leafCount = (size_t(1) << forest.GetTreeSizes()[treeIdx]);
        const auto& independentTreeShapParams = *preparedTrees->IndependentTreeShapParams;
        auto& shapValueByDepthForLeaf = preparedTrees->IndependentTreeShapParams->ShapValueByDepthBetweenLeavesForAllTrees[treeIdx];
        shapValueByDepthForLeaf.resize(leafCount);
        ////
        auto& shapValueByDepthBetweenLeaves = shapValueByDepthForLeaf[leafIdx];
        shapValueByDepthBetweenLeaves.resize(leafCount);
        CalcObliviousShapValuesByDepthForLeaf(
            forest,
            independentTreeShapParams.ReferenceLeafIndicesForAllTrees[treeIdx],
            independentTreeShapParams.BinFeatureCombinationClassByDepthForAllTrees[treeIdx],
            independentTreeShapParams.Weights,
            preparedTrees->CombinationClassFeatures.size(),
            leafIdx,
            treeIdx,
            independentTreeShapParams.IsCalcForAllLeafesForAllTrees[treeIdx],
            &shapValueByDepthBetweenLeaves
        );
    }
} 
