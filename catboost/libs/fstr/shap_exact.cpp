#include "shap_exact.h"


using namespace NCB;

static inline void CalcObliviousExactShapValuesForLeafRecursive(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    size_t documentLeafIdx,
    size_t treeIdx,
    TVector<TVector<double>>& subtreeWeights,
    const THashMap<int, size_t>& featureMap,
    TVector<double>& maskValue,
    size_t mask,
    size_t depth,
    size_t nodeIdx
) {
    const size_t approxDimension = forest.GetDimensionsCount();
    size_t treeSize = forest.GetModelTreeData()->GetTreeSizes()[treeIdx];

    if (depth == treeSize) {
        auto firstLeafPtr = forest.GetFirstLeafPtrForTree(treeIdx);
        for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
            maskValue[dimension] += subtreeWeights[depth][nodeIdx] / subtreeWeights[0][0] *
                                    firstLeafPtr[nodeIdx * approxDimension + dimension];
        }
        return;
    }

    size_t remainingDepth = treeSize - depth - 1;

    const int feature = binFeatureCombinationClass[
        forest.GetModelTreeData()->GetTreeSplits()[forest.GetModelTreeData()->GetTreeStartOffsets()[treeIdx] + remainingDepth]
    ];
    if ((mask >> featureMap.at(feature)) & 1) {
        for (size_t newDepth = depth + 1; newDepth < treeSize + 1; ++newDepth) {
            size_t numNodes = (size_t(1) << (newDepth - depth));
            size_t levelSize = (size_t(1) << (newDepth - depth - 1));
            for (size_t newNodeIdx = nodeIdx * numNodes; newNodeIdx < nodeIdx * numNodes + levelSize; ++newNodeIdx) {
                subtreeWeights[newDepth][newNodeIdx] += subtreeWeights[newDepth][newNodeIdx + levelSize];
                subtreeWeights[newDepth][newNodeIdx + levelSize] = subtreeWeights[newDepth][newNodeIdx];
            }
        }

        const bool isGoRight = (documentLeafIdx >> remainingDepth) & 1;
        const size_t goNodeIdx = nodeIdx * 2 + isGoRight;
        if (!FuzzyEquals(1 + subtreeWeights[depth + 1][goNodeIdx], 1 + 0.0)) {
            CalcObliviousExactShapValuesForLeafRecursive(
                forest,
                binFeatureCombinationClass,
                documentLeafIdx,
                treeIdx,
                subtreeWeights,
                featureMap,
                maskValue,
                mask,
                depth + 1,
                goNodeIdx
            );
        }
    } else {
        if (!FuzzyEquals(1 + subtreeWeights[depth + 1][2 * nodeIdx], 1 + 0.0)) {
            CalcObliviousExactShapValuesForLeafRecursive(
                forest,
                binFeatureCombinationClass,
                documentLeafIdx,
                treeIdx,
                subtreeWeights,
                featureMap,
                maskValue,
                mask,
                depth + 1,
                2 * nodeIdx
            );
        }
        if (!FuzzyEquals(1 + subtreeWeights[depth + 1][2 * nodeIdx + 1], 1 + 0.0)) {
            CalcObliviousExactShapValuesForLeafRecursive(
                forest,
                binFeatureCombinationClass,
                documentLeafIdx,
                treeIdx,
                subtreeWeights,
                featureMap,
                maskValue,
                mask,
                depth + 1,
                2 * nodeIdx + 1
            );
        }
    }
}

static inline TVector<int> CollectObliviousTreeFeatures(
        const TModelTrees& forest,
        const TVector<int>& binFeatureCombinationClass,
        size_t treeIdx
) {
    THashSet<int> featureSet;

    size_t treeSize = forest.GetModelTreeData()->GetTreeSizes()[treeIdx];

    for (size_t depth = 0; depth < treeSize; ++depth) {
        featureSet.insert(binFeatureCombinationClass[
                                  forest.GetModelTreeData()->GetTreeSplits()[forest.GetModelTreeData()->GetTreeStartOffsets()[treeIdx] + treeSize - depth - 1]
                          ]);
    }

    TVector<int> featureVector;
    for (int feature : featureSet) {
        featureVector.push_back(feature);
    }

    return featureVector;
}

static inline THashMap<int, size_t> ReverseFeatureVector(
        const TVector<int>& featureVector
) {
    THashMap<int, size_t> featureMap;
    for (size_t i = 0; i < featureVector.size(); ++i) {
        featureMap[featureVector[i]] = i;
    }
    return featureMap;
}

void CalcObliviousExactShapValuesForLeafImplementation(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    size_t documentLeafIdx,
    size_t treeIdx,
    const TVector<TVector<double>>& subtreeWeights,
    TVector<TShapValue>* shapValues
) {
    TVector<int> featureVector = CollectObliviousTreeFeatures(
        forest,
        binFeatureCombinationClass,
        treeIdx
    );

    const size_t approxDimension = forest.GetDimensionsCount();

    for (int feature : featureVector) {
        shapValues->emplace_back(feature, approxDimension);
    }

    THashMap<int, size_t> featureMap = ReverseFeatureVector(featureVector);

    TVector<double> coefficients(featureVector.size(), double(1) / featureVector.size());
    for (size_t i = 1; i < featureVector.size(); ++i) {
        coefficients[i] = coefficients[i - 1] * double(i) / double(featureVector.size() - i);
    }

    for (size_t mask = 0; (mask >> featureVector.size()) == 0; ++mask) {
        TVector<TVector<double>> subtreeWeightsCopy(subtreeWeights);
        TVector<double> maskValue(approxDimension, 0);
        CalcObliviousExactShapValuesForLeafRecursive(
            forest,
            binFeatureCombinationClass,
            documentLeafIdx,
            treeIdx,
            subtreeWeightsCopy,
            featureMap,
            maskValue,
            mask,
            /*depth*/ 0,
            /*nodeIdx*/ 0
        );

        size_t numUsedFeatures = 0;
        for (size_t i = 0; i < featureVector.size(); ++i) {
            numUsedFeatures += ((mask >> i) & 1);
        }

        for (size_t i = 0; i < featureVector.size(); ++i) {
            if ((mask >> i) & 1) {
                for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
                    (*shapValues)[i].Value[dimension] += coefficients[numUsedFeatures - 1] * maskValue[dimension];
                }
            } else {
                for (size_t dimension = 0; dimension < approxDimension; ++dimension) {
                    (*shapValues)[i].Value[dimension] -= coefficients[numUsedFeatures] * maskValue[dimension];
                }
            }
        }
    }
}
