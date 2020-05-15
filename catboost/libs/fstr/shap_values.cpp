#include "shap_values.h"

#include "util.h"
#include "shap_exact.h"

#include <catboost/private/libs/algo/features_data_helpers.h>
#include <catboost/private/libs/algo/index_calcer.h>
#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/algorithm.h>
#include <util/generic/cast.h>
#include <util/generic/utility.h>
#include <util/generic/ymath.h>
#include <catboost/libs/model/cpu/quantization.h>


using namespace NCB;


namespace {
    struct TFeaturePathElement {
        int Feature;
        double ZeroPathsFraction;
        double OnePathsFraction;
        double Weight;

        TFeaturePathElement() = default;

        TFeaturePathElement(int feature, double zeroPathsFraction, double onePathsFraction, double weight)
            : Feature(feature)
            , ZeroPathsFraction(zeroPathsFraction)
            , OnePathsFraction(onePathsFraction)
            , Weight(weight)
        {
        }
    };
} //anonymous

static TVector<TFeaturePathElement> ExtendFeaturePath(
    const TVector<TFeaturePathElement>& oldFeaturePath,
    double zeroPathsFraction,
    double onePathsFraction,
    int feature
) {
    const size_t pathLength = oldFeaturePath.size();

    TVector<TFeaturePathElement> newFeaturePath(pathLength + 1);
    Copy(oldFeaturePath.begin(), oldFeaturePath.begin() + pathLength, newFeaturePath.begin());

    const double weight = pathLength == 0 ? 1.0 : 0.0;
    newFeaturePath[pathLength] = TFeaturePathElement(feature, zeroPathsFraction, onePathsFraction, weight);

    for (int elementIdx = pathLength - 1; elementIdx >= 0; --elementIdx) {
        newFeaturePath[elementIdx + 1].Weight += onePathsFraction * newFeaturePath[elementIdx].Weight * (elementIdx + 1) / (pathLength + 1);
        newFeaturePath[elementIdx].Weight = zeroPathsFraction * newFeaturePath[elementIdx].Weight * (pathLength - elementIdx) / (pathLength + 1);
    }

    return newFeaturePath;
}

static TVector<TFeaturePathElement> UnwindFeaturePath(
    const TVector<TFeaturePathElement>& oldFeaturePath,
    size_t eraseElementIdx)
{
    const size_t pathLength = oldFeaturePath.size();
    CB_ENSURE(pathLength > 0, "Path to unwind must have at least one element");

    TVector<TFeaturePathElement> newFeaturePath(
        oldFeaturePath.begin(),
        oldFeaturePath.begin() + pathLength - 1);

    for (size_t elementIdx = eraseElementIdx; elementIdx < pathLength - 1; ++elementIdx) {
        newFeaturePath[elementIdx].Feature = oldFeaturePath[elementIdx + 1].Feature;
        newFeaturePath[elementIdx].ZeroPathsFraction = oldFeaturePath[elementIdx + 1].ZeroPathsFraction;
        newFeaturePath[elementIdx].OnePathsFraction = oldFeaturePath[elementIdx + 1].OnePathsFraction;
    }

    const double onePathsFraction = oldFeaturePath[eraseElementIdx].OnePathsFraction;
    const double zeroPathsFraction = oldFeaturePath[eraseElementIdx].ZeroPathsFraction;
    double weightDiff = oldFeaturePath[pathLength - 1].Weight;

    if (!FuzzyEquals(1 + onePathsFraction, 1 + 0.0)) {
        for (int elementIdx = pathLength - 2; elementIdx >= 0; --elementIdx) {
            double oldWeight = newFeaturePath[elementIdx].Weight;
            newFeaturePath[elementIdx].Weight = weightDiff * pathLength
                / (onePathsFraction * (elementIdx + 1));
            weightDiff = oldWeight
                - newFeaturePath[elementIdx].Weight * zeroPathsFraction * (pathLength - elementIdx - 1)
                    / pathLength;
        }
    } else {
        for (int elementIdx = pathLength - 2; elementIdx >= 0; --elementIdx) {
            newFeaturePath[elementIdx].Weight *= pathLength
                / (zeroPathsFraction * (pathLength - elementIdx - 1));
        }
    }

    return newFeaturePath;
}

static void UpdateShapByFeaturePath(
    const TVector<TFeaturePathElement>& featurePath,
    const double* leafValuesPtr,
    size_t leafId,
    int approxDimension,
    bool isOblivious,
    double averageTreeApprox,
    double conditionFeatureFraction,
    TVector<TShapValue>* shapValuesInternal
) {
    const int approxDimOffset = isOblivious ? approxDimension : 1;
    for (size_t elementIdx = 1; elementIdx < featurePath.size(); ++elementIdx) {
        const TVector<TFeaturePathElement> unwoundPath = UnwindFeaturePath(featurePath, elementIdx);
        double weightSum = 0.0;
        for (const TFeaturePathElement& unwoundPathElement : unwoundPath) {
            weightSum += unwoundPathElement.Weight;
        }
        const TFeaturePathElement& element = featurePath[elementIdx];
        const auto sameFeatureShapValue = FindIf(
            shapValuesInternal->begin(),
            shapValuesInternal->end(),
            [element](const TShapValue& shapValue) {
                return shapValue.Feature == element.Feature;
            }
        );
        const double coefficient =
            conditionFeatureFraction * weightSum * (element.OnePathsFraction - element.ZeroPathsFraction);
        if (sameFeatureShapValue == shapValuesInternal->end()) {
            shapValuesInternal->emplace_back(element.Feature, approxDimension);
            for (int dimension = 0; dimension < approxDimension; ++dimension) {
                double value = coefficient * (leafValuesPtr[leafId * approxDimOffset + dimension] - averageTreeApprox);
                shapValuesInternal->back().Value[dimension] = value;
            }
        } else {
            for (int dimension = 0; dimension < approxDimension; ++dimension) {
                double addValue = coefficient * (leafValuesPtr[leafId * approxDimOffset + dimension] - averageTreeApprox);
                sameFeatureShapValue->Value[dimension] += addValue;
            }
        }
    }
}

TConditionsFeatureFraction::TConditionsFeatureFraction(
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int combinationClass,
    double conditionFeatureFraction,
    double hotCoefficient,
    double coldCoefficient
) {
    HotConditionFeatureFraction = conditionFeatureFraction;
    ColdConditionFeatureFraction = conditionFeatureFraction;
    if (fixedFeatureParams.Defined() && combinationClass == fixedFeatureParams->Feature) {
        switch (fixedFeatureParams->FixedFeatureMode) {
            case TFixedFeatureParams::EMode::FixedOn: {
                ColdConditionFeatureFraction = 0;
                break;
            }
            case TFixedFeatureParams::EMode::FixedOff: {
                HotConditionFeatureFraction *= hotCoefficient;
                ColdConditionFeatureFraction *= coldCoefficient;
                break;
            }
            default: {
                Y_UNREACHABLE();
            }
        }
    }
}

static void ExtendFeaturePathIfFeatureNotFixed(
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    const TVector<TFeaturePathElement>& oldFeaturePath,
    double zeroPathsFraction,
    double onePathsFraction,
    int feature,
    TVector<TFeaturePathElement>* featurePath
) {
    if (!fixedFeatureParams.Defined() ||
        (fixedFeatureParams->FixedFeatureMode == TFixedFeatureParams::EMode::NotFixed || fixedFeatureParams->Feature != feature)) {
        *featurePath = ExtendFeaturePath(
            oldFeaturePath,
            zeroPathsFraction,
            onePathsFraction,
            feature);
    } else {
        const size_t pathLength = oldFeaturePath.size();
        *featurePath = TVector<TFeaturePathElement>(oldFeaturePath.begin(), oldFeaturePath.begin() + pathLength);
    }
}

static void CalcObliviousInternalShapValuesForLeafRecursive(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    size_t documentLeafIdx,
    size_t treeIdx,
    int depth,
    const TVector<TVector<double>>& subtreeWeights,
    size_t nodeIdx,
    const TVector<TFeaturePathElement>& oldFeaturePath,
    double zeroPathsFraction,
    double onePathsFraction,
    int feature,
    bool calcInternalValues,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    const double conditionFeatureFraction,
    TVector<TShapValue>* shapValuesInternal,
    double averageTreeApprox
) {
    if (FuzzyEquals(1.0 + conditionFeatureFraction, 1.0 + 0.0)) {
        return;
    }

    TVector<TFeaturePathElement> featurePath;
    ExtendFeaturePathIfFeatureNotFixed(
        fixedFeatureParams,
        oldFeaturePath,
        zeroPathsFraction,
        onePathsFraction,
        feature,
        &featurePath
    );

    if (depth == forest.GetTreeSizes()[treeIdx]) {
        UpdateShapByFeaturePath(
            featurePath,
            forest.GetFirstLeafPtrForTree(treeIdx),
            nodeIdx,
            forest.GetDimensionsCount(),
            /*isOblivious*/ true,
            averageTreeApprox,
            conditionFeatureFraction,
            shapValuesInternal
        );
    } else {
        double newZeroPathsFraction = 1.0;
        double newOnePathsFraction = 1.0;

        const size_t remainingDepth = forest.GetTreeSizes()[treeIdx] - depth - 1;
        const int combinationClass = binFeatureCombinationClass[
            forest.GetTreeSplits()[forest.GetTreeStartOffsets()[treeIdx] + remainingDepth]
        ];

        const auto sameFeatureElement = FindIf(
            featurePath.begin(),
            featurePath.end(),
            [combinationClass](const TFeaturePathElement& element) {
                return element.Feature == combinationClass;
            }
        );

        if (sameFeatureElement != featurePath.end()) {
            const size_t sameFeatureIndex = sameFeatureElement - featurePath.begin();
            newZeroPathsFraction = featurePath[sameFeatureIndex].ZeroPathsFraction;
            newOnePathsFraction = featurePath[sameFeatureIndex].OnePathsFraction;
            featurePath = UnwindFeaturePath(featurePath, sameFeatureIndex);
        }

        const bool isGoRight = (documentLeafIdx >> remainingDepth) & 1;
        const size_t goNodeIdx = nodeIdx * 2 + isGoRight;
        const size_t skipNodeIdx = nodeIdx * 2 + !isGoRight;
        const double hotCoefficient = subtreeWeights[depth + 1][goNodeIdx] / subtreeWeights[depth][nodeIdx];
        const double coldCoefficient = subtreeWeights[depth + 1][skipNodeIdx] / subtreeWeights[depth][nodeIdx];
        TConditionsFeatureFraction conditionsFeatureFraction{
            fixedFeatureParams,
            combinationClass,
            conditionFeatureFraction,
            hotCoefficient,
            coldCoefficient
        };

        if (!FuzzyEquals(1 + subtreeWeights[depth + 1][goNodeIdx], 1 + 0.0)) {
            double newZeroPathsFractionGoNode = newZeroPathsFraction * hotCoefficient;
            CalcObliviousInternalShapValuesForLeafRecursive(
                forest,
                binFeatureCombinationClass,
                documentLeafIdx,
                treeIdx,
                depth + 1,
                subtreeWeights,
                goNodeIdx,
                featurePath,
                newZeroPathsFractionGoNode,
                newOnePathsFraction,
                combinationClass,
                calcInternalValues,
                fixedFeatureParams,
                conditionsFeatureFraction.HotConditionFeatureFraction,
                shapValuesInternal,
                averageTreeApprox
            );
        }

        if (!FuzzyEquals(1 + subtreeWeights[depth + 1][skipNodeIdx], 1 + 0.0)) {
            double newZeroPathsFractionSkipNode = newZeroPathsFraction * coldCoefficient;
            CalcObliviousInternalShapValuesForLeafRecursive(
                forest,
                binFeatureCombinationClass,
                documentLeafIdx,
                treeIdx,
                depth + 1,
                subtreeWeights,
                skipNodeIdx,
                featurePath,
                newZeroPathsFractionSkipNode,
                /*onePathFraction*/ 0,
                combinationClass,
                calcInternalValues,
                fixedFeatureParams,
                conditionsFeatureFraction.ColdConditionFeatureFraction,
                shapValuesInternal,
                averageTreeApprox
            );
        }
    }
}

static void CalcNonObliviousInternalShapValuesForLeafRecursive(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<bool>& mapNodeIdToIsGoRight,
    size_t treeIdx,
    int depth,
    const TVector<TVector<double>>& subtreeWeights,
    size_t nodeIdx,
    const TVector<TFeaturePathElement>& oldFeaturePath,
    double zeroPathsFraction,
    double onePathsFraction,
    int feature,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    const double conditionFeatureFraction,
    bool calcInternalValues,
    TVector<TShapValue>* shapValuesInternal,
    double averageTreeApprox
) {
    if (FuzzyEquals(1.0 + conditionFeatureFraction, 1.0 + 0.0)) {
        return;
    }

    TVector<TFeaturePathElement> featurePath;
    ExtendFeaturePathIfFeatureNotFixed(
        fixedFeatureParams,
        oldFeaturePath,
        zeroPathsFraction,
        onePathsFraction,
        feature,
        &featurePath
    );

    const auto& node = forest.GetNonSymmetricStepNodes()[nodeIdx];
    const size_t startOffset = forest.GetTreeStartOffsets()[treeIdx];
    size_t goNodeIdx;
    size_t skipNodeIdx;
    if (mapNodeIdToIsGoRight[nodeIdx - startOffset]) {
        goNodeIdx = nodeIdx + node.RightSubtreeDiff;
        skipNodeIdx = nodeIdx + node.LeftSubtreeDiff;
    } else {
        goNodeIdx = nodeIdx + node.LeftSubtreeDiff;
        skipNodeIdx = nodeIdx + node.RightSubtreeDiff;
    }
    // goNodeIdx == nodeIdx mean that nodeIdx is a terminal node for
    // observed object. That's why we should update shap here.
    // Similary for skipNodeIdx.
    if (goNodeIdx == nodeIdx || skipNodeIdx == nodeIdx) {
        UpdateShapByFeaturePath(
            featurePath,
            &forest.GetLeafValues()[0],
            forest.GetNonSymmetricNodeIdToLeafId()[nodeIdx],
            forest.GetDimensionsCount(),
            /*isOblivious*/ false,
            averageTreeApprox,
            conditionFeatureFraction,
            shapValuesInternal
        );
    }
    double newZeroPathsFraction = 1.0;
    double newOnePathsFraction = 1.0;

    const int combinationClass = binFeatureCombinationClass[
        forest.GetTreeSplits()[nodeIdx]
    ];

    const auto sameFeatureElement = FindIf(
        featurePath.begin(),
        featurePath.end(),
        [combinationClass](const TFeaturePathElement& element) {
            return element.Feature == combinationClass;
        }
    );

    if (sameFeatureElement != featurePath.end()) {
        const size_t sameFeatureIndex = sameFeatureElement - featurePath.begin();
        newZeroPathsFraction = featurePath[sameFeatureIndex].ZeroPathsFraction;
        newOnePathsFraction = featurePath[sameFeatureIndex].OnePathsFraction;
        featurePath = UnwindFeaturePath(featurePath, sameFeatureIndex);
    }

    const double hotCoefficient = goNodeIdx != nodeIdx ?
        subtreeWeights[0][goNodeIdx - startOffset] / subtreeWeights[0][nodeIdx - startOffset] : -1.0;
    const double coldCoefficient = skipNodeIdx != nodeIdx ?
        subtreeWeights[0][skipNodeIdx - startOffset] / subtreeWeights[0][nodeIdx - startOffset] : -1.0;
    TConditionsFeatureFraction conditionsFeatureFraction {
        fixedFeatureParams,
        combinationClass,
        conditionFeatureFraction,
        hotCoefficient,
        coldCoefficient
    };

    if (goNodeIdx != nodeIdx && !FuzzyEquals(1 + subtreeWeights[0][goNodeIdx - startOffset], 1 + 0.0)) {
        double newZeroPathsFractionGoNode = newZeroPathsFraction * hotCoefficient;
        CalcNonObliviousInternalShapValuesForLeafRecursive(
            forest,
            binFeatureCombinationClass,
            mapNodeIdToIsGoRight,
            treeIdx,
            depth + 1,
            subtreeWeights,
            goNodeIdx,
            featurePath,
            newZeroPathsFractionGoNode,
            newOnePathsFraction,
            combinationClass,
            fixedFeatureParams,
            conditionsFeatureFraction.HotConditionFeatureFraction,
            calcInternalValues,
            shapValuesInternal,
            averageTreeApprox
        );
    }

    if (skipNodeIdx != nodeIdx && !FuzzyEquals(1 + subtreeWeights[0][skipNodeIdx - startOffset], 1 + 0.0)) {
        double newZeroPathsFractionSkipNode = newZeroPathsFraction * coldCoefficient;
        CalcNonObliviousInternalShapValuesForLeafRecursive(
            forest,
            binFeatureCombinationClass,
            mapNodeIdToIsGoRight,
            treeIdx,
            depth + 1,
            subtreeWeights,
            skipNodeIdx,
            featurePath,
            newZeroPathsFractionSkipNode,
            /*onePathFraction*/ 0,
            combinationClass,
            fixedFeatureParams,
            conditionsFeatureFraction.ColdConditionFeatureFraction,
            calcInternalValues,
            shapValuesInternal,
            averageTreeApprox
        );
    }
}

static void UnpackInternalShaps(const TVector<TShapValue>& shapValuesInternal, const TVector<TVector<int>>& combinationClassFeatures,  TVector<TShapValue>* shapValues) {
    shapValues->clear();
    if (shapValuesInternal.empty()) {
        return;
    }
    const int approxDimension = shapValuesInternal[0].Value.ysize();
    for (const auto & shapValueInternal: shapValuesInternal) {
        const TVector<int> &flatFeatures = combinationClassFeatures[shapValueInternal.Feature];

        for (int flatFeatureIdx : flatFeatures) {
            const auto sameFeatureShapValue = FindIf(
                shapValues->begin(),
                shapValues->end(),
                [flatFeatureIdx](const TShapValue &shapValue) {
                    return shapValue.Feature == flatFeatureIdx;
                }
            );
            double coefficient = flatFeatures.size();
            if (sameFeatureShapValue == shapValues->end()) {
                shapValues->emplace_back(flatFeatureIdx, approxDimension);
                for (int dimension = 0; dimension < approxDimension; ++dimension) {
                    double value = shapValueInternal.Value[dimension] / coefficient;
                    shapValues->back().Value[dimension] = value;
                }
            } else {
                for (int dimension = 0; dimension < approxDimension; ++dimension) {
                    double addValue = shapValueInternal.Value[dimension] / coefficient;
                    sameFeatureShapValue->Value[dimension] += addValue;
                }
            }
        }
    }
}

static inline void CalcObliviousShapValuesForLeaf(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    size_t documentLeafIdx,
    size_t treeIdx,
    const TVector<TVector<double>>& subtreeWeights,
    bool calcInternalValues,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    TVector<TShapValue>* shapValues,
    double averageTreeApprox
) {
    shapValues->clear();

    if (calcInternalValues) {
        CalcObliviousInternalShapValuesForLeafRecursive(
            forest,
            binFeatureCombinationClass,
            documentLeafIdx,
            treeIdx,
            /*depth*/ 0,
            subtreeWeights,
            /*nodeIdx*/ 0,
            /*initialFeaturePath*/ {},
            /*zeroPathFraction*/ 1,
            /*onePathFraction*/ 1,
            /*feature*/ -1,
            calcInternalValues,
            fixedFeatureParams,
            /*conditionFeatureFraction*/ 1,
            shapValues,
            averageTreeApprox
        );
    } else {
        TVector<TShapValue> shapValuesInternal;
        CalcObliviousInternalShapValuesForLeafRecursive(
            forest,
            binFeatureCombinationClass,
            documentLeafIdx,
            treeIdx,
            /*depth*/ 0,
            subtreeWeights,
            /*nodeIdx*/ 0,
            /*initialFeaturePath*/ {},
            /*zeroPathFraction*/ 1,
            /*onePathFraction*/ 1,
            /*feature*/ -1,
            calcInternalValues,
            fixedFeatureParams,
            /*conditionFeatureFraction*/ 1,
            &shapValuesInternal,
            averageTreeApprox
        );
        UnpackInternalShaps(shapValuesInternal, combinationClassFeatures, shapValues);
    }
}

static void CalcObliviousApproximateShapValuesForLeafImplementation(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    size_t documentLeafIdx,
    size_t treeIdx,
    const TVector<TVector<TVector<double>>>& subtreeValues,
    TVector<TShapValue>* shapValues
) {
    const size_t approxDimension = forest.GetDimensionsCount();
    size_t treeSize = forest.GetTreeSizes()[treeIdx];
    size_t nodeIdx = 0;
    for (size_t depth = 0; depth < treeSize; ++depth) {
        size_t remainingDepth = treeSize - depth - 1;
        const bool isGoRight = (documentLeafIdx >> remainingDepth) & 1;
        const size_t goNodeIdx = nodeIdx * 2 + isGoRight;
        const int combinationClass = binFeatureCombinationClass[
            forest.GetTreeSplits()[forest.GetTreeStartOffsets()[treeIdx] + remainingDepth]
        ];
        const auto FeatureShapValue = FindIf(
            shapValues->begin(),
            shapValues->end(),
            [combinationClass](const TShapValue &shapValue) {
                return shapValue.Feature == combinationClass;
            }
        );
        auto newFeatureShapValue = shapValues->end();
        if (FeatureShapValue == shapValues->end()) {
            shapValues->emplace_back(combinationClass, approxDimension);
            newFeatureShapValue = shapValues->end() - 1;
        } else {
            newFeatureShapValue = FeatureShapValue;
        }
        for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
            newFeatureShapValue->Value[dimension] +=
                subtreeValues[depth + 1][goNodeIdx][dimension]
                - subtreeValues[depth][nodeIdx][dimension];
        }
        nodeIdx = goNodeIdx;
    }
}

static inline void CalcObliviousApproximateShapValuesForLeaf(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    size_t documentLeafIdx,
    size_t treeIdx,
    const TVector<TVector<TVector<double>>>& subtreeValues,
    bool calcInternalValues,
    TVector<TShapValue>* shapValues
) {
   shapValues->clear();

   if (calcInternalValues) {
        CalcObliviousApproximateShapValuesForLeafImplementation(
            forest,
            binFeatureCombinationClass,
            documentLeafIdx,
            treeIdx,
            subtreeValues,
            shapValues
        );
   } else {
       TVector<TShapValue> shapValuesInternal;
       CalcObliviousApproximateShapValuesForLeafImplementation(
           forest,
           binFeatureCombinationClass,
           documentLeafIdx,
           treeIdx,
           subtreeValues,
           &shapValuesInternal
       );
       UnpackInternalShaps(shapValuesInternal, combinationClassFeatures, shapValues);
   }
}

static inline void CalcObliviousExactShapValuesForLeaf(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    size_t documentLeafIdx,
    size_t treeIdx,
    const TVector<TVector<double>>& subtreeWeights,
    bool calcInternalValues,
    TVector<TShapValue>* shapValues
) {
    shapValues->clear();

    if (calcInternalValues) {
        CalcObliviousExactShapValuesForLeafImplementation(
            forest,
            binFeatureCombinationClass,
            documentLeafIdx,
            treeIdx,
            subtreeWeights,
            shapValues
        );
    } else {
        TVector<TShapValue> shapValuesInternal;
        CalcObliviousExactShapValuesForLeafImplementation(
            forest,
            binFeatureCombinationClass,
            documentLeafIdx,
            treeIdx,
            subtreeWeights,
            &shapValuesInternal
        );
        UnpackInternalShaps(shapValuesInternal, combinationClassFeatures, shapValues);
    }
}

static inline void CalcNonObliviousShapValuesForLeaf(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    const TVector<bool>& mapNodeIdToIsGoRight,
    size_t treeIdx,
    const TVector<TVector<double>>& subtreeWeights,
    bool calcInternalValues,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    TVector<TShapValue>* shapValues,
    double averageTreeApprox
) {
    shapValues->clear();

    if (calcInternalValues) {
        CalcNonObliviousInternalShapValuesForLeafRecursive(
            forest,
            binFeatureCombinationClass,
            mapNodeIdToIsGoRight,
            treeIdx,
            /*depth*/ 0,
            subtreeWeights,
            /*nodeIdx*/ forest.GetTreeStartOffsets()[treeIdx],
            /*initialFeaturePath*/ {},
            /*zeroPathFraction*/ 1,
            /*onePathFraction*/ 1,
            /*feature*/ -1,
            fixedFeatureParams,
            /*conditionFeatureFraction*/ 1.0,
            calcInternalValues,
            shapValues,
            averageTreeApprox
        );
    } else {
        TVector<TShapValue> shapValuesInternal;
        CalcNonObliviousInternalShapValuesForLeafRecursive(
            forest,
            binFeatureCombinationClass,
            mapNodeIdToIsGoRight,
            treeIdx,
            /*depth*/ 0,
            subtreeWeights,
            /*nodeIdx*/ forest.GetTreeStartOffsets()[treeIdx],
            /*initialFeaturePath*/ {},
            /*zeroPathFraction*/ 1,
            /*onePathFraction*/ 1,
            /*feature*/ -1,
            fixedFeatureParams,
            /*conditionFeatureFraction*/ 1.0,
            calcInternalValues,
            &shapValuesInternal,
            averageTreeApprox
        );
        UnpackInternalShaps(shapValuesInternal, combinationClassFeatures, shapValues);
    }
}

static void CalcNonObliviousApproximateShapValuesForLeafImplementation(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<bool>& mapNodeIdToIsGoRight,
    size_t treeIdx,
    const TVector<TVector<TVector<double>>>& subtreeValues,
    TVector<TShapValue>* shapValues
) {
    const size_t approxDimension = forest.GetDimensionsCount();
    const size_t startOffset = forest.GetTreeStartOffsets()[treeIdx];
    size_t nodeIdx = startOffset;
    size_t goNodeIdx;
    auto& node = forest.GetNonSymmetricStepNodes()[nodeIdx];
    if (mapNodeIdToIsGoRight[nodeIdx - startOffset]) {
        goNodeIdx = nodeIdx + node.RightSubtreeDiff;
    } else {
        goNodeIdx = nodeIdx + node.LeftSubtreeDiff;
    }
    while (nodeIdx != goNodeIdx) {
        const int combinationClass = binFeatureCombinationClass[
            forest.GetTreeSplits()[nodeIdx]
        ];
        const auto FeatureShapValue = FindIf(
            shapValues->begin(),
            shapValues->end(),
            [combinationClass](const TShapValue &shapValue) {
                return shapValue.Feature == combinationClass;
            }
        );
        auto newFeatureShapValue = shapValues->end();
        if (FeatureShapValue == shapValues->end()) {
            shapValues->emplace_back(combinationClass, approxDimension);
            newFeatureShapValue = shapValues->end() - 1;
        } else {
            newFeatureShapValue = FeatureShapValue;
        }
        for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
            newFeatureShapValue->Value[dimension] +=
                subtreeValues[0][goNodeIdx - startOffset][dimension]
                - subtreeValues[0][nodeIdx - startOffset][dimension];
        }
        nodeIdx = goNodeIdx;
        auto& node = forest.GetNonSymmetricStepNodes()[nodeIdx];
        if (mapNodeIdToIsGoRight[nodeIdx - startOffset]) {
            goNodeIdx = nodeIdx + node.RightSubtreeDiff;
        } else {
            goNodeIdx = nodeIdx + node.LeftSubtreeDiff;
        }
    }
}

static inline void CalcNonObliviousApproximateShapValuesForLeaf(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    const TVector<bool>& mapNodeIdToIsGoRight,
    size_t treeIdx,
    const TVector<TVector<TVector<double>>>& subtreeValues,
    bool calcInternalValues,
    TVector<TShapValue>* shapValues
) {
    shapValues->clear();

    if (calcInternalValues) {
        CalcNonObliviousApproximateShapValuesForLeafImplementation(
            forest,
            binFeatureCombinationClass,
            mapNodeIdToIsGoRight,
            treeIdx,
            subtreeValues,
            shapValues
        );
    } else {
        TVector<TShapValue> shapValuesInternal;
        CalcNonObliviousApproximateShapValuesForLeafImplementation(
            forest,
            binFeatureCombinationClass,
            mapNodeIdToIsGoRight,
            treeIdx,
            subtreeValues,
            &shapValuesInternal
        );
        UnpackInternalShaps(shapValuesInternal, combinationClassFeatures, shapValues);
    }
}

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

// All calculations only for one docId
static TVector<bool> GetDocumentPathToLeafForNonObliviousBlock(
    const TModelTrees& forest,
    const size_t docIdx,
    const size_t treeIdx,
    const NCB::NModelEvaluation::TCPUEvaluatorQuantizedData& block
) {
    const ui8* binFeatures = block.QuantizedData.data();
    const size_t docCountInBlock = block.ObjectsCount;
    const TRepackedBin* treeSplitsPtr = forest.GetRepackedBins().data();
    const auto firstLeafOffsets = forest.GetFirstLeafOffsets();
    const int totalNodesCount = forest.GetTreeSplits().size();
    const bool isLastTree = static_cast<size_t>(treeIdx + 1) == forest.GetTreeStartOffsets().size();
    const size_t endOffset = isLastTree ? totalNodesCount : forest.GetTreeStartOffsets()[treeIdx + 1];
    TVector<bool> mapNodeIdToIsGoRight;
    for (NCB::NModelEvaluation::TCalcerIndexType nodeIdx = forest.GetTreeStartOffsets()[treeIdx]; nodeIdx < endOffset; ++nodeIdx) {
        const TRepackedBin split = treeSplitsPtr[nodeIdx];
        ui8 featureValue = binFeatures[split.FeatureIndex * docCountInBlock + docIdx];
        if (!forest.GetOneHotFeatures().empty()) {
            featureValue ^= split.XorMask;
        }
        mapNodeIdToIsGoRight.push_back(featureValue >= split.SplitIdx);
    }
    return mapNodeIdToIsGoRight;
}

static TVector<bool> GetDocumentIsGoRightMapperForNodesInNonObliviousTree(
    const TModelTrees& forest,
    size_t treeIdx,
    const NCB::NModelEvaluation::IQuantizedData* binarizedFeaturesForBlock,
    size_t documentIdx
) {
    const NModelEvaluation::TCPUEvaluatorQuantizedData* dataPtr = reinterpret_cast<const NModelEvaluation::TCPUEvaluatorQuantizedData*>(binarizedFeaturesForBlock);
    Y_ASSERT(dataPtr);
    auto blockId = documentIdx / NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE;
    auto subBlock = dataPtr->ExtractBlock(blockId);
    return GetDocumentPathToLeafForNonObliviousBlock(
        forest,
        documentIdx % NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE,
        treeIdx,
        subBlock
    );
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
        const size_t treeDepth = forest.GetTreeSizes()[treeIdx];
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
        const size_t startOffset = forest.GetTreeStartOffsets()[treeIdx];
        auto firstLeafPtr = &forest.GetLeafValues()[0];
        TVector<size_t> reversedTree = GetReversedSubtreeForNonObliviousTree(forest, treeIdx);
        subtreeValues.resize(1);
        subtreeValues[0].resize(reversedTree.size(), TVector<double>(approxDimension, 0.0));
        if (reversedTree.size() == 1) {
            size_t leafIdx = forest.GetNonSymmetricNodeIdToLeafId()[startOffset];
            for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
                subtreeValues[0][0][dimension] = firstLeafPtr[leafIdx + dimension];
            }
        } else {
            for (size_t localIdx = reversedTree.size() - 1; localIdx > 0; --localIdx) {
                size_t leafIdx = forest.GetNonSymmetricNodeIdToLeafId()[startOffset + localIdx];
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

void CalcShapValuesForDocumentMulti(
    const TFullModel& model,
    const TShapPreparedTrees& preparedTrees,
    const NCB::NModelEvaluation::IQuantizedData* binarizedFeaturesForBlock,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int featuresCount,
    TConstArrayRef<NModelEvaluation::TCalcerIndexType> docIndices,
    size_t documentIdxInBlock,
    TVector<TVector<double>>* shapValues,
    ECalcTypeShapValues calcType
) {
    const int approxDimension = model.GetDimensionsCount();
    shapValues->assign(approxDimension, TVector<double>(featuresCount + 1, 0.0));
    const size_t treeCount = model.GetTreeCount();
    for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        if (preparedTrees.CalcShapValuesByLeafForAllTrees && model.IsOblivious()) {
            Y_ASSERT(docIndices[treeIdx] < preparedTrees.ShapValuesByLeafForAllTrees[treeIdx].size());
            for (const TShapValue& shapValue : preparedTrees.ShapValuesByLeafForAllTrees[treeIdx][docIndices[treeIdx]]) {
                for (int dimension = 0; dimension < approxDimension; ++dimension) {
                    (*shapValues)[dimension][shapValue.Feature] += shapValue.Value[dimension];
                }
            }
        } else {
            TVector<TShapValue> shapValuesByLeaf;
            switch (calcType) {
                case ECalcTypeShapValues::Approximate:
                    if (model.IsOblivious()) {
                        CalcObliviousApproximateShapValuesForLeaf(
                            *model.ModelTrees.Get(),
                            preparedTrees.BinFeatureCombinationClass,
                            preparedTrees.CombinationClassFeatures,
                            docIndices[treeIdx],
                            treeIdx,
                            preparedTrees.SubtreeValuesForAllTrees[treeIdx],
                            preparedTrees.CalcInternalValues,
                            &shapValuesByLeaf
                        );
                    } else {
                        TVector<bool> mapNodeIdToIsGoRight = GetDocumentIsGoRightMapperForNodesInNonObliviousTree(
                            *model.ModelTrees.Get(),
                            treeIdx,
                            binarizedFeaturesForBlock,
                            documentIdxInBlock
                        );
                        CalcNonObliviousApproximateShapValuesForLeaf(
                            *model.ModelTrees.Get(),
                            preparedTrees.BinFeatureCombinationClass,
                            preparedTrees.CombinationClassFeatures,
                            mapNodeIdToIsGoRight,
                            treeIdx,
                            preparedTrees.SubtreeValuesForAllTrees[treeIdx],
                            preparedTrees.CalcInternalValues,
                            &shapValuesByLeaf
                        );
                    }
                    break;
                case ECalcTypeShapValues::Regular:
                    if (model.IsOblivious()) {
                        CalcObliviousShapValuesForLeaf(
                            *model.ModelTrees.Get(),
                            preparedTrees.BinFeatureCombinationClass,
                            preparedTrees.CombinationClassFeatures,
                            docIndices[treeIdx],
                            treeIdx,
                            preparedTrees.SubtreeWeightsForAllTrees[treeIdx],
                            preparedTrees.CalcInternalValues,
                            fixedFeatureParams,
                            &shapValuesByLeaf,
                            preparedTrees.AverageApproxByTree[treeIdx]
                        );
                    } else {
                        TVector<bool> mapNodeIdToIsGoRight = GetDocumentIsGoRightMapperForNodesInNonObliviousTree(
                            *model.ModelTrees.Get(),
                            treeIdx,
                            binarizedFeaturesForBlock,
                            documentIdxInBlock
                        );
                        CalcNonObliviousShapValuesForLeaf(
                            *model.ModelTrees.Get(),
                            preparedTrees.BinFeatureCombinationClass,
                            preparedTrees.CombinationClassFeatures,
                            mapNodeIdToIsGoRight,
                            treeIdx,
                            preparedTrees.SubtreeWeightsForAllTrees[treeIdx],
                            preparedTrees.CalcInternalValues,
                            fixedFeatureParams,
                            &shapValuesByLeaf,
                            preparedTrees.AverageApproxByTree[treeIdx]
                        );
                    }
                    break;
                case ECalcTypeShapValues::Exact:
                    CB_ENSURE(model.IsOblivious(), "'Exact' calculation type is supported only for symmetric trees.");
                    CalcObliviousExactShapValuesForLeaf(
                        *model.ModelTrees.Get(),
                        preparedTrees.BinFeatureCombinationClass,
                        preparedTrees.CombinationClassFeatures,
                        docIndices[treeIdx],
                        treeIdx,
                        preparedTrees.SubtreeWeightsForAllTrees[treeIdx],
                        preparedTrees.CalcInternalValues,
                        &shapValuesByLeaf
                    );
                    break;
            }

            for (const TShapValue& shapValue : shapValuesByLeaf) {
                for (int dimension = 0; dimension < approxDimension; ++dimension) {
                    (*shapValues)[dimension][shapValue.Feature] += shapValue.Value[dimension];
                }
            }
        }

        for (int dimension = 0; dimension < approxDimension; ++dimension) {
            (*shapValues)[dimension][featuresCount] +=
                    preparedTrees.MeanValuesForAllTrees[treeIdx][dimension];
        }
    }
    if (approxDimension == 1) {
        (*shapValues)[0][featuresCount] += model.GetScaleAndBias().Bias;
    }
}

void CalcShapValuesForDocumentMulti(
    const TFullModel& model,
    const TShapPreparedTrees& preparedTrees,
    const NCB::NModelEvaluation::IQuantizedData* binarizedFeaturesForBlock,
    int featuresCount,
    TConstArrayRef<NModelEvaluation::TCalcerIndexType> docIndices,
    size_t documentIdxInBlock,
    TVector<TVector<double>>* shapValues,
    ECalcTypeShapValues calcType
) {
    CalcShapValuesForDocumentMulti(
        model,
        preparedTrees,
        binarizedFeaturesForBlock,
        /*fixedFeatureParams*/ Nothing(),
        featuresCount,
        docIndices,
        documentIdxInBlock,
        shapValues,
        calcType
    );
}

static void CalcShapValuesForDocumentBlockMulti(
    const TFullModel& model,
    const IFeaturesBlockIterator& featuresBlockIterator,
    int flatFeatureCount,
    const TShapPreparedTrees& preparedTrees,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    size_t start,
    size_t end,
    NPar::TLocalExecutor* localExecutor,
    TVector<TVector<TVector<double>>>* shapValuesForAllDocuments,
    ECalcTypeShapValues calcType
) {
    CheckNonZeroApproxForZeroWeightLeaf(model);

    const size_t documentCount = end - start;

    auto binarizedFeaturesForBlock = MakeQuantizedFeaturesForEvaluator(model, featuresBlockIterator, start, end);

    TVector<NModelEvaluation::TCalcerIndexType> indices(binarizedFeaturesForBlock->GetObjectsCount() * model.GetTreeCount());
    model.GetCurrentEvaluator()->CalcLeafIndexes(binarizedFeaturesForBlock.Get(), 0, model.GetTreeCount(), indices);

    const int oldShapValuesSize = shapValuesForAllDocuments->size();
    shapValuesForAllDocuments->resize(oldShapValuesSize + end - start);

    NPar::TLocalExecutor::TExecRangeParams blockParams(0, documentCount);
    localExecutor->ExecRange([&] (size_t documentIdxInBlock) {
        TVector<TVector<double>>& shapValues = (*shapValuesForAllDocuments)[oldShapValuesSize + documentIdxInBlock];

        CalcShapValuesForDocumentMulti(
            model,
            preparedTrees,
            binarizedFeaturesForBlock.Get(),
            fixedFeatureParams,
            flatFeatureCount,
            MakeArrayRef(indices.data() + documentIdxInBlock * model.GetTreeCount(), model.GetTreeCount()),
            documentIdxInBlock,
            &shapValues,
            calcType
        );

    }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
}

static double CalcAverageApprox(const TVector<double>& averageApproxByClass) {
    double result = 0;
    for (double value : averageApproxByClass) {
        result += value;
    }
    return result / averageApproxByClass.size();
}

static void CalcShapValuesByLeafForTreeBlock(
    const TModelTrees& forest,
    int start,
    int end,
    bool calcInternalValues,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    NPar::TLocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees,
    ECalcTypeShapValues calcType
) {
    const auto& binFeatureCombinationClass = preparedTrees->BinFeatureCombinationClass;
    const auto& combinationClassFeatures = preparedTrees->CombinationClassFeatures;

    NPar::TLocalExecutor::TExecRangeParams blockParams(start, end);
    localExecutor->ExecRange([&] (size_t treeIdx) {
        const bool isOblivious = forest.GetNonSymmetricStepNodes().empty() && forest.GetNonSymmetricNodeIdToLeafId().empty();
        if (preparedTrees->CalcShapValuesByLeafForAllTrees && isOblivious) {
            const size_t leafCount = (size_t(1) << forest.GetTreeSizes()[treeIdx]);
            TVector<TVector<TShapValue>>& shapValuesByLeaf = preparedTrees->ShapValuesByLeafForAllTrees[treeIdx];
            shapValuesByLeaf.resize(leafCount);
            for (size_t leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
                switch (calcType) {
                    case ECalcTypeShapValues::Approximate:
                        CalcObliviousApproximateShapValuesForLeaf(
                            forest,
                            binFeatureCombinationClass,
                            combinationClassFeatures,
                            leafIdx,
                            treeIdx,
                            preparedTrees->SubtreeValuesForAllTrees[treeIdx],
                            calcInternalValues,
                            &shapValuesByLeaf[leafIdx]
                        );
                        break;
                    case ECalcTypeShapValues::Regular:
                        CalcObliviousShapValuesForLeaf(
                            forest,
                            binFeatureCombinationClass,
                            combinationClassFeatures,
                            leafIdx,
                            treeIdx,
                            preparedTrees->SubtreeWeightsForAllTrees[treeIdx],
                            calcInternalValues,
                            fixedFeatureParams,
                            &shapValuesByLeaf[leafIdx],
                            preparedTrees->AverageApproxByTree[treeIdx]
                        );
                        break;
                    case ECalcTypeShapValues::Exact:
                        CalcObliviousExactShapValuesForLeaf(
                            forest,
                            binFeatureCombinationClass,
                            combinationClassFeatures,
                            leafIdx,
                            treeIdx,
                            preparedTrees->SubtreeWeightsForAllTrees[treeIdx],
                            calcInternalValues,
                            &shapValuesByLeaf[leafIdx]
                        );
                        break;
                }
            }
        }
    }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
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
    TShapPreparedTrees* preparedTrees,
    ECalcTypeShapValues calcType
) {
    const size_t treeCount = forest.GetTreeCount();
    for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
        preparedTrees->SubtreeWeightsForAllTrees[treeIdx] = CalcSubtreeWeightsForTree(forest, leafWeights, treeIdx);
        preparedTrees->MeanValuesForAllTrees[treeIdx]
           = CalcMeanValueForTree(forest, preparedTrees->SubtreeWeightsForAllTrees[treeIdx], treeIdx);
        if (calcType == ECalcTypeShapValues::Approximate) {
            preparedTrees->SubtreeValuesForAllTrees[treeIdx] =
                    CalcSubtreeValuesForTree(forest, preparedTrees->SubtreeWeightsForAllTrees[treeIdx],
                                             leafWeights, treeIdx);
        }
        preparedTrees->AverageApproxByTree[treeIdx] = isMultiClass ? CalcAverageApprox(preparedTrees->MeanValuesForAllTrees[treeIdx]) : 0;
    }
}

static void InitPreparedTrees(
    const TFullModel& model,
    const TDataProvider* dataset, // can be nullptr if model has LeafWeights
    EPreCalcShapValues mode,
    bool calcInternalValues,
    NPar::TLocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees,
    ECalcTypeShapValues calcType
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
    if (calcType == ECalcTypeShapValues::Approximate) {
        preparedTrees->SubtreeValuesForAllTrees.resize(treeCount);
    }
    preparedTrees->AverageApproxByTree.resize(treeCount);
    preparedTrees->CalcInternalValues = calcInternalValues;

    const TModelTrees& forest = *model.ModelTrees;
    MapBinFeaturesToClasses(
        forest,
        &preparedTrees->BinFeatureCombinationClass,
        &preparedTrees->CombinationClassFeatures
    );
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

void CalcShapValuesByLeaf(
    const TFullModel& model,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int logPeriod,
    bool calcInternalValues,
    NPar::TLocalExecutor* localExecutor,
    TShapPreparedTrees* preparedTrees,
    ECalcTypeShapValues calcType
) {
    const size_t treeCount = model.GetTreeCount();
    const size_t treeBlockSize = CB_THREAD_LIMIT; // least necessary for threading
    TProfileInfo processTreesProfile(treeCount);
    TImportanceLogger treesLogger(treeCount, "trees processed", "Processing trees...", logPeriod);

    for (size_t start = 0; start < treeCount; start += treeBlockSize) {
        size_t end = Min(start + treeBlockSize, treeCount);

        processTreesProfile.StartIterationBlock();

        CalcShapValuesByLeafForTreeBlock(
            *model.ModelTrees,
            start,
            end,
            calcInternalValues,
            fixedFeatureParams,
            localExecutor,
            preparedTrees,
            calcType
        );

        processTreesProfile.FinishIterationBlock(end - start);
        auto profileResults = processTreesProfile.GetProfileResults();
        treesLogger.Log(profileResults);
    }
}

TShapPreparedTrees PrepareTrees(
    const TFullModel& model,
    const TDataProvider* dataset, // can be nullptr if model has LeafWeights
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor,
    bool calcInternalValues,
    ECalcTypeShapValues calcType
) {
    TVector<double> leafWeights;
    InitLeafWeights(model, dataset, localExecutor, &leafWeights);
    TShapPreparedTrees preparedTrees;
    InitPreparedTrees(model, dataset, mode, calcInternalValues, localExecutor, &preparedTrees, calcType);
    const bool isMultiClass = IsMultiClassification(model);
    CalcTreeStats(*model.ModelTrees, leafWeights, isMultiClass, &preparedTrees, calcType);
    return preparedTrees;
}

TShapPreparedTrees PrepareTrees(
    const TFullModel& model,
    NPar::TLocalExecutor* localExecutor,
    ECalcTypeShapValues calcType
) {
    CB_ENSURE(
        !model.ModelTrees->GetLeafWeights().empty(),
        "Model must have leaf weights or sample pool must be provided"
    );
    TShapPreparedTrees preparedTrees = PrepareTrees(
        model,
        nullptr,
        EPreCalcShapValues::Auto, localExecutor,
        /*calcInternalValues*/ false,
        calcType
    );
    CalcShapValuesByLeaf(
        model,
        /*fixedFeatureParams*/ Nothing(),
        /*logPeriod*/ 0,
        preparedTrees.CalcInternalValues,
        localExecutor,
        &preparedTrees,
        calcType
    );
    return preparedTrees;
}

void CalcShapValuesInternalForFeature(
    const TShapPreparedTrees& preparedTrees,
    const TFullModel& model,
    int /*logPeriod*/,
    ui32 start,
    ui32 end,
    ui32 featuresCount,
    const NCB::TObjectsDataProvider& objectsData,
    TVector<TVector<TVector<double>>>* shapValues, // [docIdx][featureIdx][dim]
    NPar::TLocalExecutor* localExecutor,
    ECalcTypeShapValues calcType
) {

    CB_ENSURE(start <= end && end <= objectsData.GetObjectCount());
    const TModelTrees& forest = *model.ModelTrees;
    shapValues->clear();
    const ui32 documentCount = end - start;
    shapValues->resize(documentCount);

    THolder<IFeaturesBlockIterator> featuresBlockIterator
        = CreateFeaturesBlockIterator(model, objectsData, start, end);

    const ui32 documentBlockSize = NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE;
    TVector<NModelEvaluation::TCalcerIndexType> indices(documentBlockSize * forest.GetTreeCount());

    for (ui32 startIdx = 0; startIdx < documentCount; startIdx += documentBlockSize) {
        NPar::TLocalExecutor::TExecRangeParams blockParams(startIdx, startIdx + Min(documentBlockSize, documentCount - startIdx));
        featuresBlockIterator->NextBlock(blockParams.LastId - blockParams.FirstId);
        auto binarizedFeaturesForBlock = MakeQuantizedFeaturesForEvaluator(model, *featuresBlockIterator, blockParams.FirstId, blockParams.LastId);

        model.GetCurrentEvaluator()->CalcLeafIndexes(
            binarizedFeaturesForBlock.Get(),
            0, forest.GetTreeCount(),
            MakeArrayRef(indices.data(), binarizedFeaturesForBlock->GetObjectsCount() * forest.GetTreeCount())
        );

        localExecutor->ExecRange([&](ui32 documentIdx) {
            TVector<TVector<double>> &docShapValues = (*shapValues)[documentIdx];
            docShapValues.assign(featuresCount, TVector<double>(forest.GetDimensionsCount() + 1, 0.0));
            auto docIndices = MakeArrayRef(indices.data() + forest.GetTreeCount() * (documentIdx - startIdx), forest.GetTreeCount());
            for (size_t treeIdx = 0; treeIdx < forest.GetTreeCount(); ++treeIdx) {
                if (preparedTrees.CalcShapValuesByLeafForAllTrees && model.IsOblivious()) {
                    for (const TShapValue& shapValue : preparedTrees.ShapValuesByLeafForAllTrees[treeIdx][docIndices[treeIdx]]) {
                        for (int dimension = 0; dimension < (int)forest.GetDimensionsCount(); ++dimension) {
                            docShapValues[shapValue.Feature][dimension] += shapValue.Value[dimension];
                        }
                    }
                } else {
                    TVector<TShapValue> shapValuesByLeaf;
                    switch (calcType) {
                        case ECalcTypeShapValues::Approximate:
                            if (model.IsOblivious()) {
                                CalcObliviousApproximateShapValuesForLeaf(
                                    forest,
                                    preparedTrees.BinFeatureCombinationClass,
                                    preparedTrees.CombinationClassFeatures,
                                    docIndices[treeIdx],
                                    treeIdx,
                                    preparedTrees.SubtreeValuesForAllTrees[treeIdx],
                                    preparedTrees.CalcInternalValues,
                                    &shapValuesByLeaf
                                );
                            } else {
                                const TVector<bool> docPathIndexes = GetDocumentIsGoRightMapperForNodesInNonObliviousTree(
                                    *model.ModelTrees.Get(),
                                    treeIdx,
                                    binarizedFeaturesForBlock.Get(),
                                    documentIdx - startIdx
                                );
                                CalcNonObliviousApproximateShapValuesForLeaf(
                                    forest,
                                    preparedTrees.BinFeatureCombinationClass,
                                    preparedTrees.CombinationClassFeatures,
                                    docPathIndexes,
                                    treeIdx,
                                    preparedTrees.SubtreeValuesForAllTrees[treeIdx],
                                    preparedTrees.CalcInternalValues,
                                    &shapValuesByLeaf
                                );
                            }
                            break;
                        case ECalcTypeShapValues::Regular:
                            if (model.IsOblivious()) {
                                CalcObliviousShapValuesForLeaf(
                                    forest,
                                    preparedTrees.BinFeatureCombinationClass,
                                    preparedTrees.CombinationClassFeatures,
                                    docIndices[treeIdx],
                                    treeIdx,
                                    preparedTrees.SubtreeWeightsForAllTrees[treeIdx],
                                    preparedTrees.CalcInternalValues,
                                    /*fixedFeatureParams*/ Nothing(),
                                    &shapValuesByLeaf,
                                    preparedTrees.AverageApproxByTree[treeIdx]
                                );
                            } else {
                                const TVector<bool> docPathIndexes = GetDocumentIsGoRightMapperForNodesInNonObliviousTree(
                                    *model.ModelTrees.Get(),
                                    treeIdx,
                                    binarizedFeaturesForBlock.Get(),
                                    documentIdx - startIdx
                                );
                                CalcNonObliviousShapValuesForLeaf(
                                    forest,
                                    preparedTrees.BinFeatureCombinationClass,
                                    preparedTrees.CombinationClassFeatures,
                                    docPathIndexes,
                                    treeIdx,
                                    preparedTrees.SubtreeWeightsForAllTrees[treeIdx],
                                    preparedTrees.CalcInternalValues,
                                    /*fixedFeatureParams*/ Nothing(),
                                    &shapValuesByLeaf,
                                    preparedTrees.AverageApproxByTree[treeIdx]
                                );
                            }
                            break;
                        case ECalcTypeShapValues::Exact:
                            CB_ENSURE(model.IsOblivious(), "'Exact' calculation type is supported only for symmetric trees.");
                            CalcObliviousExactShapValuesForLeaf(
                                forest,
                                preparedTrees.BinFeatureCombinationClass,
                                preparedTrees.CombinationClassFeatures,
                                docIndices[treeIdx],
                                treeIdx,
                                preparedTrees.SubtreeWeightsForAllTrees[treeIdx],
                                preparedTrees.CalcInternalValues,
                                &shapValuesByLeaf
                            );
                            break;
                    }

                    for (const TShapValue& shapValue : shapValuesByLeaf) {
                        for (int dimension = 0; dimension < (int)forest.GetDimensionsCount(); ++dimension) {
                            docShapValues[shapValue.Feature][dimension] += shapValue.Value[dimension];
                        }
                    }
                }
            }
        }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
    }
}

static TVector<TVector<TVector<double>>> CalcShapValuesWithPreparedTrees(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int logPeriod,
    TShapPreparedTrees* preparedTrees,
    NPar::TLocalExecutor* localExecutor,
    ECalcTypeShapValues calcType
) {
    const size_t documentCount = dataset.ObjectsGrouping->GetObjectCount();
    const size_t documentBlockSize = CB_THREAD_LIMIT; // least necessary for threading

    const int flatFeatureCount = SafeIntegerCast<int>(dataset.MetaInfo.GetFeatureCount());

    TImportanceLogger documentsLogger(documentCount, "documents processed", "Processing documents...", logPeriod);

    TVector<TVector<TVector<double>>> shapValues;
    shapValues.reserve(documentCount);

    TProfileInfo processDocumentsProfile(documentCount);

    THolder<IFeaturesBlockIterator> featuresBlockIterator
        = CreateFeaturesBlockIterator(model, *dataset.ObjectsData, 0, documentCount);

    for (size_t start = 0; start < documentCount; start += documentBlockSize) {
        size_t end = Min(start + documentBlockSize, documentCount);

        processDocumentsProfile.StartIterationBlock();

        featuresBlockIterator->NextBlock(end - start);

        CalcShapValuesForDocumentBlockMulti(
            model,
            *featuresBlockIterator,
            flatFeatureCount,
            *preparedTrees,
            fixedFeatureParams,
            start,
            end,
            localExecutor,
            &shapValues,
            calcType
        );

        processDocumentsProfile.FinishIterationBlock(end - start);
        auto profileResults = processDocumentsProfile.GetProfileResults();
        documentsLogger.Log(profileResults);
    }

    return shapValues;
}

TVector<TVector<TVector<double>>> CalcShapValuesMulti(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor,
    ECalcTypeShapValues calcType
) {
    TShapPreparedTrees preparedTrees = PrepareTrees(
        model,
        &dataset,
        mode,
        localExecutor,
        /*calcInternalValues*/ false,
        calcType
    );
    CalcShapValuesByLeaf(
        model,
        fixedFeatureParams,
        logPeriod,
        preparedTrees.CalcInternalValues,
        localExecutor,
        &preparedTrees,
        calcType
    );

    return CalcShapValuesWithPreparedTrees(
        model,
        dataset,
        fixedFeatureParams,
        logPeriod,
        &preparedTrees,
        localExecutor,
        calcType
    );
}

TVector<TVector<double>> CalcShapValues(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor,
    ECalcTypeShapValues calcType
) {
    CB_ENSURE(model.ModelTrees->GetDimensionsCount() == 1, "Model must not be trained for multiclassification.");
    TVector<TVector<TVector<double>>> shapValuesMulti = CalcShapValuesMulti(
        model,
        dataset,
        fixedFeatureParams,
        logPeriod,
        mode,
        localExecutor,
        calcType
    );

    size_t documentsCount = dataset.ObjectsGrouping->GetObjectCount();
    TVector<TVector<double>> shapValues(documentsCount);

    for (size_t documentIdx = 0; documentIdx < documentsCount; ++documentIdx) {
        shapValues[documentIdx] = std::move(shapValuesMulti[documentIdx][0]);
    }
    return shapValues;
}

static TVector<TVector<TVector<double>>> SwapFeatureAndDocumentAxes(const TVector<TVector<TVector<double>>>& shapValues) {
    const size_t approxDimension = shapValues[0].size();
    const size_t documentCount = shapValues.size();
    const size_t featuresCount = shapValues[0][0].size();
    TVector<TVector<TVector<double>>> swapedShapValues(featuresCount);
    for (size_t featureIdx = 0; featureIdx < featuresCount; ++featureIdx) {
        swapedShapValues[featureIdx].resize(approxDimension);
        for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
            swapedShapValues[featureIdx][dimension].resize(documentCount);
            for (size_t documentIdx = 0; documentIdx < documentCount; ++documentIdx) {
                swapedShapValues[featureIdx][dimension][documentIdx] = shapValues[documentIdx][dimension][featureIdx];
            }
        }
    }
    return swapedShapValues;
}

// returned: ShapValues[featureIdx][dim][documentIdx]
TVector<TVector<TVector<double>>> CalcShapValueWithQuantizedData(
    const TFullModel& model,
    const TVector<TIntrusivePtr<NModelEvaluation::IQuantizedData>>& quantizedFeatures,
    const TVector<TVector<NModelEvaluation::TCalcerIndexType>>& indices,
    const TMaybe<TFixedFeatureParams>& fixedFeatureParams,
    const size_t documentCount,
    int logPeriod,
    TShapPreparedTrees* preparedTrees,
    NPar::TLocalExecutor* localExecutor,
    ECalcTypeShapValues calcType
) {
    CalcShapValuesByLeaf(
        model,
        fixedFeatureParams,
        logPeriod,
        preparedTrees->CalcInternalValues,
        localExecutor,
        preparedTrees,
        calcType
    );
    const TModelTrees& forest = *model.ModelTrees;
    TVector<TVector<TVector<double>>> shapValues(documentCount);
    const int featuresCount = preparedTrees->CombinationClassFeatures.size();
    const size_t documentBlockSize = CB_THREAD_LIMIT;
    for (ui32 startIdx = 0, blockIdx = 0; startIdx < documentCount; startIdx += documentBlockSize, ++blockIdx) {
        NPar::TLocalExecutor::TExecRangeParams blockParams(startIdx, startIdx + Min(documentBlockSize, documentCount - startIdx));
        auto quantizedFeaturesBlock = quantizedFeatures[blockIdx];
        auto& indicesForBlock = indices[blockIdx];
        localExecutor->ExecRange([&](ui32 documentIdx) {
            const size_t documentIdxInBlock = documentIdx - startIdx;
            auto docIndices = MakeArrayRef(indicesForBlock.data() + forest.GetTreeCount() * documentIdxInBlock, forest.GetTreeCount());
            CalcShapValuesForDocumentMulti(
                model,
                *preparedTrees,
                quantizedFeaturesBlock.Get(),
                fixedFeatureParams,
                featuresCount,
                docIndices,
                documentIdxInBlock,
                &shapValues[documentIdx],
                calcType
            );
        }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
    }

    const auto& swapedShapValues = SwapFeatureAndDocumentAxes(shapValues);
    return swapedShapValues;
}

static void OutputShapValuesMulti(const TVector<TVector<TVector<double>>>& shapValues, TFileOutput& out) {
    for (const auto& shapValuesForDocument : shapValues) {
        for (const auto& shapValuesForClass : shapValuesForDocument) {
            int valuesCount = shapValuesForClass.size();
            for (int valueIdx = 0; valueIdx < valuesCount; ++valueIdx) {
                out << shapValuesForClass[valueIdx] << (valueIdx + 1 == valuesCount ? '\n' : '\t');
            }
        }
    }
}

void CalcAndOutputShapValues(
    const TFullModel& model,
    const TDataProvider& dataset,
    const TString& outputPath,
    int logPeriod,
    EPreCalcShapValues mode,
    NPar::TLocalExecutor* localExecutor,
    ECalcTypeShapValues calcType
) {
    TShapPreparedTrees preparedTrees = PrepareTrees(
        model,
        &dataset,
        mode,
        localExecutor,
        /*calcInternalValues*/ false,
        calcType
    );
    CalcShapValuesByLeaf(
        model,
        /*fixedFeatureParams*/ Nothing(),
        logPeriod,
        preparedTrees.CalcInternalValues,
        localExecutor,
        &preparedTrees,
        calcType
    );

    CB_ENSURE_SCALE_IDENTITY(model.GetScaleAndBias(), "SHAP values");
    const int flatFeatureCount = SafeIntegerCast<int>(dataset.MetaInfo.GetFeatureCount());

    const size_t documentCount = dataset.ObjectsGrouping->GetObjectCount();
    const size_t documentBlockSize = CB_THREAD_LIMIT; // least necessary for threading

    TImportanceLogger documentsLogger(documentCount, "documents processed", "Processing documents...", logPeriod);

    TProfileInfo processDocumentsProfile(documentCount);

    THolder<IFeaturesBlockIterator> featuresBlockIterator
        = CreateFeaturesBlockIterator(model, *dataset.ObjectsData, 0, documentCount);

    TFileOutput out(outputPath);
    for (size_t start = 0; start < documentCount; start += documentBlockSize) {
        size_t end = Min(start + documentBlockSize, documentCount);
        processDocumentsProfile.StartIterationBlock();

        TVector<TVector<TVector<double>>> shapValuesForBlock;
        shapValuesForBlock.reserve(end - start);

        featuresBlockIterator->NextBlock(end - start);

        CalcShapValuesForDocumentBlockMulti(
            model,
            *featuresBlockIterator,
            flatFeatureCount,
            preparedTrees,
            /*fixedFeatureParams*/ Nothing(),
            start,
            end,
            localExecutor,
            &shapValuesForBlock,
            calcType
        );

        OutputShapValuesMulti(shapValuesForBlock, out);

        processDocumentsProfile.FinishIterationBlock(end - start);
        auto profileResults = processDocumentsProfile.GetProfileResults();
        documentsLogger.Log(profileResults);
    }
}
