#include "independent_tree_shap.h"

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


namespace {
    struct TContribution {
        TVector<double> PositiveContribution;
        TVector<double> NegativeContribution;

    public:

        explicit TContribution(size_t approxDimension)
            : PositiveContribution(approxDimension)
            , NegativeContribution(approxDimension)
            {
            }
    };

    class TInternalIndependentTreeShapCalcer {
    private:
        const TVector<int>& BinFeatureCombinationClassByDepth;
        const TVector<TVector<double>>& Weights;
        TVector<int> ListOfFeaturesDocumentLeaf;
        TVector<int> ListOfFeaturesDocumentLeafReference;
        size_t DocumentLeafIdx;
        size_t DocumentLeafIdxReference;
        int DepthOfTree;
        size_t ApproxDimension;
        const double* LeafValuesPtr;
        TVector<TVector<double>>& ShapValuesInternalByDepth;

    public:
        TInternalIndependentTreeShapCalcer(
            const TModelTrees& forest,
            const TVector<int>& binFeatureCombinationClassByDepth,
            const TVector<TVector<double>>& weights,
            size_t classCount,
            size_t documentLeafIdx,
            size_t documentLeafIdxReference,
            size_t treeIdx,
            TVector<TVector<double>>* shapValuesInternalByDepth
        )
            : BinFeatureCombinationClassByDepth(binFeatureCombinationClassByDepth) 
            , Weights(weights) 
            , ListOfFeaturesDocumentLeaf(classCount) 
            , ListOfFeaturesDocumentLeafReference(classCount) 
            , DocumentLeafIdx(documentLeafIdx) 
            , DocumentLeafIdxReference(documentLeafIdxReference) 
            , DepthOfTree(forest.GetTreeSizes()[treeIdx]) 
            , ApproxDimension(forest.GetDimensionsCount()) 
            , LeafValuesPtr(forest.GetFirstLeafPtrForTree(treeIdx)) 
            , ShapValuesInternalByDepth(*shapValuesInternalByDepth) 
        { 
        }

        TContribution Calc(
            int depth = 0,
            size_t nodeIdx = 0,
            ui32 uniqueFeaturesCount = 0,
            ui32 featureMatchedForegroundCount = 0
        );
    };
} //anonymous


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

static void AddValuesToShapValuesByReference(
    const TVector<TVector<double>>& shapValueByDepthForLeaf,
    const TVector<int>& binFeatureCombinationClassByDepth,
    TVector<TVector<double>>* shapValuesByReference
) {
    for (size_t dimension = 0; dimension < shapValueByDepthForLeaf.size(); ++dimension) {
        TConstArrayRef<double> shapValueByDepthForLeafRef = MakeConstArrayRef(shapValueByDepthForLeaf[dimension]);
        TArrayRef<double> shapValuesByReferenceRef = MakeArrayRef((*shapValuesByReference)[dimension]);
        for (int depth = 0; depth < (int)shapValueByDepthForLeafRef.size() - 1; ++depth) {
            const auto featureIdx = binFeatureCombinationClassByDepth[depth];
            shapValuesByReferenceRef[featureIdx] += shapValueByDepthForLeafRef[depth];
        }
        // add mean values
        shapValuesByReferenceRef.back() += shapValueByDepthForLeafRef.back();
    }
}

void AddValuesToShapValuesByAllReferences(
    const TVector<TVector<TVector<double>>>& shapValueByDepthForLeaf,
    const TVector<NCB::NModelEvaluation::TCalcerIndexType>& referenceLeafIndices,
    const TVector<int>& binFeatureCombinationClassByDepth,
    TVector<TVector<TVector<double>>>* shapValuesForAllReferences
) {
    for (size_t referenceIdx = 0; referenceIdx < referenceLeafIndices.size(); ++referenceIdx) {
        AddValuesToShapValuesByReference(
            shapValueByDepthForLeaf[referenceLeafIndices[referenceIdx]],
            binFeatureCombinationClassByDepth,
            &shapValuesForAllReferences->at(referenceIdx)
        );
    }
}

static inline ui64 GetBinomialCoeffient(ui64 n, ui64 k) { 
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

/*
static void UnpackInternalShapsByDepth(
    const TVector<TVector<double>>& shapValuesInternal, // [depth][dim]
    const TVector<TVector<int>>& combinationClassFeatures,
    const TVector<int>& binFeatureCombinationClassByDepth,
    TVector<TVector<double>>* shapValues
) {
    if (shapValuesInternal.empty()) {
        return;
    }
    const size_t depthOfTree = shapValuesInternal.size() - 1;
    const size_t approxDimension = shapValuesInternal[0].size();
    for (size_t depth = 0; depth < depthOfTree; ++depth) {
        const int combinationClass = binFeatureCombinationClassByDepth[depth];
        const TVector<int> &flatFeatures = combinationClassFeatures[combinationClass];
        TConstArrayRef<double> shapValuesInternalByClass = MakeConstArrayRef(shapValuesInternal[depth]);
        double coefficient = flatFeatures.size();
        for (int flatFeatureIdx : flatFeatures) {
            TArrayRef<double> shapValuesByFeature = MakeArrayRef((*shapValues)[flatFeatureIdx]);
            for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
                double addValue = shapValuesInternalByClass[dimension] / coefficient;
                shapValuesByFeature[dimension] += addValue;
            }   
        }
    }
}

static void UnpackShapsByDepth(
    const TVector<TVector<double>>& shapValuesInternalByDepth, // [depth][dim]
    const TVector<int>& binFeatureCombinationClassByDepth,
    TVector<TVector<double>>* shapValuesInternal
) {
    if (shapValuesInternalByDepth.empty()) {
        return;
    }
    const size_t depthOfTree = shapValuesInternalByDepth.size() - 1;
    for (size_t depth = 0; depth < depthOfTree; ++depth) {
        const int combinationClass = binFeatureCombinationClassByDepth[depth];
        (*shapValuesInternal)[combinationClass] = shapValuesInternalByDepth[depth];
    }
}


static TVector<TVector<double>> GetShapValuesByFeatures(
    const TVector<TVector<double>>& shapValuesInternalByDepth, // [depth][dim]
    const TVector<TVector<int>>& combinationClassFeatures,
    const TVector<int>& binFeatureCombinationClassByDepth,
    int flatFeatureCount,
    bool calcInternalValues
) {
    TVector<TVector<double>> shapValues;
    if (shapValuesInternalByDepth.empty()){
        return shapValues;
    }
    const size_t approxDimension = shapValuesInternalByDepth[0].size();
    if (calcInternalValues) {
        const size_t classCount = combinationClassFeatures.size();
        shapValues.assign(classCount + 1, TVector<double>(approxDimension, 0.0));
        UnpackShapsByDepth(
            shapValuesInternalByDepth,
            binFeatureCombinationClassByDepth,
            &shapValues
        );
    } else {
        shapValues.assign(flatFeatureCount + 1, TVector<double>(approxDimension, 0.0));
        UnpackInternalShapsByDepth(
            shapValuesInternalByDepth,
            combinationClassFeatures,
            binFeatureCombinationClassByDepth,
            &shapValues
        );
    }
    // set mean values
    shapValues.back() = shapValuesInternalByDepth.back();
    return shapValues;
}
*/

static TVector<TVector<double>> SwapFeatureAndDimensionAxes(const TVector<TVector<double>>& shapValues) {
    const size_t featureCount = shapValues.size();
    const size_t approxDimension = shapValues[0].size();
    TVector<TVector<double>> swapedShapValues(approxDimension);
    for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
        swapedShapValues[dimension].resize(featureCount);
        for (size_t featureIdx = 0; featureIdx < featureCount; ++featureIdx) {
            swapedShapValues[dimension][featureIdx] = shapValues[featureIdx][dimension];
        }
    }
    return swapedShapValues;
}

void CalcObliviousShapValuesByDepthForLeaf(
    const TModelTrees& forest,
    const TVector<NCB::NModelEvaluation::TCalcerIndexType>& referenceLeafIndices,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    const TVector<TVector<double>>& weights,
    size_t documentLeafIdx,
    size_t treeIdx,
    bool isCalcForAllLeafes,
    TVector<TVector<TVector<double>>>* shapValueByDepthBetweenLeaves
) {
    const auto& binFeatureCombinationClassByDepth =
        GetBinFeatureCombinationClassByDepth(forest, binFeatureCombinationClass, treeIdx);
    const size_t depthOfTree = forest.GetTreeSizes()[treeIdx];
    const size_t approxDimension = forest.GetDimensionsCount();
    const size_t leafCountInTree = (size_t(1) << forest.GetTreeSizes()[treeIdx]);
    const size_t leafCount = isCalcForAllLeafes ? leafCountInTree : referenceLeafIndices.size(); 
    const size_t classCount = combinationClassFeatures.size();
    for (size_t idx = 0; idx < leafCount; ++idx) {
        const size_t leafIdx = isCalcForAllLeafes ? idx : referenceLeafIndices[idx];
        TVector<TVector<double>> shapValueInternalBetweenLeaves(depthOfTree + 1, TVector<double>(approxDimension, 0.0));
        TInternalIndependentTreeShapCalcer calcerIntenalShaps{
            forest,
            binFeatureCombinationClassByDepth,
            weights,
            classCount,
            documentLeafIdx,
            /*documentLeafIdxReference*/ leafIdx,
            treeIdx,
            &shapValueInternalBetweenLeaves
        };
        calcerIntenalShaps.Calc();

        shapValueByDepthBetweenLeaves->at(leafIdx) = SwapFeatureAndDimensionAxes(shapValueInternalBetweenLeaves);
    }
}

static inline double TransformDocument(
    const THolder<IMetric>& metric,
    double target,
    double approx
) {
    TVector<TVector<double>> approxVector(1, TVector<double>(1, approx));
    TVector<float> targetVector(1, target);
    auto score = metric->Eval(approxVector, targetVector, {}, {}, 0, 1, NPar::LocalExecutor());
    return metric->GetFinalError(score);
}

static inline double GetTransformData(
    const THolder<IMetric>& metric,
    EModelOutputType modelOutputType,
    double target,
    double approx
) {
    switch (modelOutputType) {
        case EModelOutputType::Probability:
            return 1.0 / (1 + exp(-approx));
        case EModelOutputType::LossFunction:
            return TransformDocument(metric, target, approx);
        default:
            CB_ENSURE_INTERNAL(false, "Not recognized type of model output");
    }
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

void PostProcessingIndependent(
    const TIndependentTreeShapParams& independentTreeShapParams,
    const TVector<TVector<TVector<double>>>& shapValuesForAllReferences,
    const TVector<TVector<int>>& combinationClassFeatures,
    size_t approxDimension,
    size_t flatFeatureCount,
    size_t documentIdx,
    bool calcInternalValues,
    double bias,
    TVector<TVector<double>>* shapValues  
) {
    const size_t featureCount = calcInternalValues ? combinationClassFeatures.size() : flatFeatureCount;
    const size_t referenceCount = independentTreeShapParams.ReferenceLeafIndicesForAllTrees[0].size();
    EModelOutputType modelOutputType = independentTreeShapParams.ModelOutputType;
    const bool isNotRawOutputType = (EModelOutputType::Raw != modelOutputType);
    const auto& metric = independentTreeShapParams.Metric;
    const auto& approxOfDataset = independentTreeShapParams.ApproxOfDataset;
    const auto& approxOfReferenceDataset = independentTreeShapParams.ApproxOfReferenceDataset;
    const auto& targetOfDataset = independentTreeShapParams.TargetOfDataset;
    const auto& transformedTargetOfDataset = independentTreeShapParams.TransformedTargetOfDataset;
    for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
        TConstArrayRef<double> approxOfReferenceDatasetRef = MakeConstArrayRef(approxOfReferenceDataset[dimension]); 
        TArrayRef<double> shapValuesRef = MakeArrayRef((*shapValues)[dimension]);
        const double approxOfDocument = approxOfDataset[dimension][documentIdx];
        const double targetOfDocument = targetOfDataset[dimension][documentIdx];
        const double transformedTargetOfDocument = transformedTargetOfDataset[dimension][documentIdx];
        for (size_t referenceIdx = 0; referenceIdx < referenceCount; ++referenceIdx) {
            TVector<double> shapValueOneDimension;
            if (calcInternalValues) {
                shapValueOneDimension = std::move(shapValuesForAllReferences[referenceIdx][dimension]);
            } else {
                shapValueOneDimension.resize(flatFeatureCount + 1);
                UnpackInternalShapsForDocumentOneDimension(
                    shapValuesForAllReferences[referenceIdx][dimension],
                    combinationClassFeatures,
                    &shapValueOneDimension
                );
                // set mean value
                shapValueOneDimension.back() = shapValuesForAllReferences[referenceIdx][dimension].back();
            }
            double rescaleCoefficient = referenceCount;
            double transformedCoeffient = 1.0;
            if (isNotRawOutputType && approxOfDocument != approxOfReferenceDatasetRef[referenceIdx]) {
                double transformReferenceTarget = GetTransformData(
                    metric,
                    modelOutputType,
                    targetOfDocument,
                    approxOfReferenceDatasetRef[referenceIdx]
                );
                transformedCoeffient = (transformedTargetOfDocument - transformReferenceTarget) /
                    (approxOfDocument - approxOfReferenceDatasetRef[referenceIdx]);
            }
            double totalCoeffient = transformedCoeffient / rescaleCoefficient;
            for (size_t featureIdx = 0; featureIdx < featureCount; ++featureIdx) {
                shapValuesRef[featureIdx] += shapValueOneDimension[featureIdx] * totalCoeffient;
            }
            if (isNotRawOutputType) {
                shapValuesRef[featureCount] += GetTransformData(
                    metric,
                    modelOutputType,
                    targetOfDocument,
                    shapValueOneDimension[featureCount] + bias
                ) / rescaleCoefficient;
            } else {
                shapValuesRef[featureCount] += (shapValueOneDimension[featureCount] / rescaleCoefficient);
            }
        }
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
            double coefficient = k * GetBinomialCoeffient(k - 1, n);
            double value = 1.0 / coefficient;
            weights[n][k] = value;
        }
    }
    return weights;
}

static TVector<TVector<double>> GetTransformedTarget(
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& targetData,
    const THolder<IMetric>& metric
) {
    CB_ENSURE_INTERNAL(
        approx.size() == targetData.size() && approx[0].size() == targetData[0].size(),
        "Approx and target must have same sizes"
    );
    TVector<TVector<double>> transformedTarget(approx.size(), TVector<double>(approx[0].size(), 0.0));
    for (auto dimension : xrange(approx.size())) {
        TConstArrayRef<double> targetDataRef = MakeConstArrayRef(targetData[dimension]);
        TConstArrayRef<double> approxRef = MakeConstArrayRef(approx[dimension]);
        TArrayRef<double> transformedTargetRef = MakeArrayRef(transformedTarget[dimension]);

        for (auto documentIdx : xrange(approxRef.size())) {        
            transformedTargetRef[documentIdx] = TransformDocument(
                metric,
                targetDataRef[documentIdx],
                approxRef[documentIdx]
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
    switch (ModelOutputType)
    {
        case EModelOutputType::Probability: {
            TransformedTargetOfDataset = ApplyModelMulti(model, *dataset.ObjectsData, EPredictionType::Probability, 0, 0, localExecutor);
            break;
        }
        case EModelOutputType::LossFunction : {
            Metric = std::move(CreateMetricFromDescription(metricDescription, model.GetDimensionsCount())[0]);
            CB_ENSURE(model.GetDimensionsCount() <= 1, "Multi-dimensional target fstr is unimplemented yet");
            TransformedTargetOfDataset = GetTransformedTarget(
                ApproxOfDataset,
                TargetOfDataset,
                Metric
            );
            break;
        }
        default:
            CB_ENSURE_INTERNAL(false, "Unexpected model output type for transforming data");
    }
}

TIndependentTreeShapParams::TIndependentTreeShapParams(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TDataProvider& referenceDataset,
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
    ShapValueByDepthBetweenLeavesForAllTrees.resize(treeCount);
    for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        const size_t leafCount = size_t(1) << forest.GetTreeSizes()[treeIdx];
        ReferenceIndicesForAllTrees[treeIdx].resize(leafCount);
        ShapValueByDepthBetweenLeavesForAllTrees[treeIdx].resize(leafCount);
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

    FlatFeatureCount = SafeIntegerCast<int>(dataset.MetaInfo.GetFeatureCount());
    ApproxOfDataset = ApplyModelMulti(model, *dataset.ObjectsData, EPredictionType::RawFormulaVal, 0, 0, localExecutor);
    ApproxOfReferenceDataset = ApplyModelMulti(model, *referenceDataset.ObjectsData, EPredictionType::RawFormulaVal, 0, 0, localExecutor);
    auto targetDataProvider =
        CreateModelCompatibleProcessedDataProvider(dataset, {metricDescription}, model, GetMonopolisticFreeCpuRam(), nullptr, localExecutor).TargetData;
    CB_ENSURE(targetDataProvider->GetTarget(), "Label must be provided");
    auto targetOfDatasetRef = *(targetDataProvider->GetTarget());
    TargetOfDataset.resize(targetOfDatasetRef.size());
    for (size_t dimension = 0; dimension < targetOfDatasetRef.size(); ++dimension) {
        TargetOfDataset[dimension].resize(targetOfDatasetRef[dimension].size());
        for (size_t documentIdx = 0; documentIdx < targetOfDatasetRef[dimension].size(); ++documentIdx) {
            TargetOfDataset[dimension][documentIdx] = targetOfDatasetRef[dimension][documentIdx];
        }
    }
    if (modelOutputType != EModelOutputType::Raw) {
        InitTransformedData(model, dataset, metricDescription, localExecutor);
    }
}

void CalcIndependentTreeShapValuesByLeafForTreeBlock(
    const TModelTrees& forest,
    const TIndependentTreeShapParams& independentTreeShapParams,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    size_t treeIdx,
    TVector<TVector<TVector<TVector<double>>>>* shapValueByDepthForLeaf
) {
    const size_t leafCount = (size_t(1) << forest.GetTreeSizes()[treeIdx]);
    for (size_t leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
        auto& shapValueByDepthBetweenLeaves = (*shapValueByDepthForLeaf)[leafIdx];
        shapValueByDepthBetweenLeaves.resize(leafCount);
        CalcObliviousShapValuesByDepthForLeaf(
            forest,
            independentTreeShapParams.ReferenceLeafIndicesForAllTrees[treeIdx],
            binFeatureCombinationClass,
            combinationClassFeatures,
            independentTreeShapParams.Weights,
            leafIdx,
            treeIdx,
            independentTreeShapParams.IsCalcForAllLeafesForAllTrees[treeIdx],
            &shapValueByDepthBetweenLeaves
        );
    }
} 
