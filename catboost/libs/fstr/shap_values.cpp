#include "shap_values.h"

#include <catboost/libs/algo/index_calcer.h>

#include <util/generic/algorithm.h>

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

static TVector<TFeaturePathElement> ExtendFeaturePath(const TVector<TFeaturePathElement>& oldFeaturePath,
                                                      double zeroPathsFraction,
                                                      double onePathsFraction,
                                                      int feature) {
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

static TVector<TFeaturePathElement> UnwindFeaturePath(const TVector<TFeaturePathElement>& oldFeaturePath, size_t eraseElementIdx) {
    const size_t pathLength = oldFeaturePath.size();
    CB_ENSURE(pathLength > 0, "Path to unwind must have at least one element");

    TVector<TFeaturePathElement> newFeaturePath(oldFeaturePath.begin(), oldFeaturePath.begin() + pathLength - 1);

    for (size_t elementIdx = eraseElementIdx; elementIdx < pathLength - 1; ++elementIdx) {
        newFeaturePath[elementIdx].Feature = oldFeaturePath[elementIdx + 1].Feature;
        newFeaturePath[elementIdx].ZeroPathsFraction = oldFeaturePath[elementIdx + 1].ZeroPathsFraction;
        newFeaturePath[elementIdx].OnePathsFraction = oldFeaturePath[elementIdx + 1].OnePathsFraction;
    }

    const double onePathsFraction = oldFeaturePath[eraseElementIdx].OnePathsFraction;
    const double zeroPathsFraction = oldFeaturePath[eraseElementIdx].ZeroPathsFraction;
    double weightDiff = oldFeaturePath[pathLength - 1].Weight;

    if (!FuzzyEquals(onePathsFraction, 0.0)) {
        for (int elementIdx = pathLength - 2; elementIdx >= 0; --elementIdx) {
            double oldWeight = newFeaturePath[elementIdx].Weight;
            newFeaturePath[elementIdx].Weight = weightDiff * pathLength / (onePathsFraction * (elementIdx + 1));
            weightDiff = oldWeight - newFeaturePath[elementIdx].Weight * zeroPathsFraction * (pathLength - elementIdx - 1) / pathLength;
        }
    } else {
        for (int elementIdx = pathLength - 2; elementIdx >= 0; --elementIdx) {
            newFeaturePath[elementIdx].Weight *= pathLength / (zeroPathsFraction * (pathLength - elementIdx - 1));
        }
    }


    return newFeaturePath;
}

static void CalcShapValuesRecursive(const TObliviousTrees& forest,
                                    const TVector<int>& binFeaturesMapping,
                                    const TVector<ui8>& binFeaturesValues,
                                    size_t treeIdx,
                                    int depth,
                                    const TVector<TVector<size_t>>& subtreeSizes,
                                    int dimension,
                                    size_t nodeIdx,
                                    const TVector<TFeaturePathElement>& oldFeaturePath,
                                    double zeroPathsFraction,
                                    double onePathsFraction,
                                    int feature,
                                    TVector<double>* shapValuesPtr) {
    TVector<double>& shapValues = *shapValuesPtr;
    TVector<TFeaturePathElement> featurePath = ExtendFeaturePath(oldFeaturePath, zeroPathsFraction, onePathsFraction, feature);

    if (depth == forest.TreeSizes[treeIdx]) {
        for (size_t elementIdx = 1; elementIdx < featurePath.size(); ++elementIdx) {
            TVector<TFeaturePathElement> unwoundPath = UnwindFeaturePath(featurePath, elementIdx);
            double weightSum = 0.0;
            for (const TFeaturePathElement& unwoundPathElement : unwoundPath) {
                weightSum += unwoundPathElement.Weight;
            }
            const TFeaturePathElement& element = featurePath[elementIdx];
            const int approxDimension = forest.ApproxDimension;

            shapValues[element.Feature] += weightSum * (element.OnePathsFraction - element.ZeroPathsFraction)
                                         * forest.LeafValues[treeIdx][nodeIdx * approxDimension + dimension];
        }
    } else {
        const TRepackedBin& split = forest.GetRepackedBins()[forest.TreeStartOffsets[treeIdx] + depth];
        const int splitBinFeature = split.FeatureIndex;
        const int splitFlatFeature = binFeaturesMapping[splitBinFeature];
        const ui8 threshold = split.SplitIdx;
        const ui8 xorMask = split.XorMask;

        double newZeroPathsFraction = 1.0;
        double newOnePathsFraction = 1.0;

        const auto sameFeatureElement = FindIf(featurePath.begin(), featurePath.end(), [splitFlatFeature](const TFeaturePathElement& element) {return element.Feature == splitFlatFeature;});

        if (sameFeatureElement != featurePath.end()) {
            const size_t sameFeatureIndex = sameFeatureElement - featurePath.begin();
            newZeroPathsFraction = featurePath[sameFeatureIndex].ZeroPathsFraction;
            newOnePathsFraction = featurePath[sameFeatureIndex].OnePathsFraction;
            featurePath = UnwindFeaturePath(featurePath, sameFeatureIndex);
        }

        const size_t goNodeIdx = nodeIdx | (((binFeaturesValues[splitBinFeature] ^ xorMask) >= threshold) << depth);
        const size_t skipNodeIdx = goNodeIdx ^ (1 << depth);

        if (subtreeSizes[depth + 1][goNodeIdx] > 0) {
            CalcShapValuesRecursive(forest, binFeaturesMapping, binFeaturesValues, treeIdx, depth + 1, subtreeSizes, dimension,
                                    goNodeIdx, featurePath,
                                    newZeroPathsFraction * subtreeSizes[depth + 1][goNodeIdx] / subtreeSizes[depth][nodeIdx],
                                    newOnePathsFraction, splitFlatFeature,
                                    &shapValues);
        }

        if (subtreeSizes[depth + 1][skipNodeIdx] > 0) {
            CalcShapValuesRecursive(forest, binFeaturesMapping, binFeaturesValues, treeIdx, depth + 1, subtreeSizes, dimension,
                                    skipNodeIdx, featurePath,
                                    newZeroPathsFraction * subtreeSizes[depth + 1][skipNodeIdx] / subtreeSizes[depth][nodeIdx],
                                    /*onePathFraction*/ 0, splitFlatFeature,
                                    &shapValues);
        }
    }
}

static double CalcMeanValueForTree(const TObliviousTrees& forest,
                                   const TVector<TVector<size_t>>& subtreeSizes,
                                   size_t treeIdx,
                                   int dimension) {
    double meanValue = 0.0;

    const size_t maxDepth = forest.TreeSizes[treeIdx];

    for (size_t leafIdx = 0; leafIdx < (size_t(1) << maxDepth); ++leafIdx) {
        const int approxDimension = forest.ApproxDimension;

        meanValue += forest.LeafValues[treeIdx][leafIdx * approxDimension + dimension] * subtreeSizes[maxDepth][leafIdx];
    }

    meanValue /= subtreeSizes[0][0];

    return meanValue;
}

static TVector<TVector<size_t>> CalcSubtreeSizesForTree(const TObliviousTrees& forest, size_t treeIdx) {
    const int maxDepth = forest.TreeSizes[treeIdx];
    TVector<TVector<size_t>> subtreeSizes(maxDepth + 1);

    subtreeSizes[maxDepth].resize(size_t(1) << maxDepth);
    for (size_t leafIdx = 0; leafIdx < (size_t(1) << maxDepth); ++leafIdx) {
        subtreeSizes[maxDepth][leafIdx] = size_t(forest.LeafWeights[treeIdx][leafIdx] + 0.5);
    }

    for (int depth = maxDepth - 1; depth >= 0; --depth) {
        const size_t nodeCount = size_t(1) << depth;
        subtreeSizes[depth].resize(nodeCount);
        for (size_t nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx) {
            subtreeSizes[depth][nodeIdx] = subtreeSizes[depth + 1][nodeIdx] + subtreeSizes[depth + 1][nodeIdx ^ (1 << depth)];
        }
    }
    return subtreeSizes;
}

static TVector<int> MapFeatures(const TObliviousTrees& forest) {
    const int featureCount = forest.GetEffectiveBinaryFeaturesBucketsCount();

    TVector<int> flatFeatureIndices;
    flatFeatureIndices.reserve(featureCount);

    for (const TFloatFeature& floatFeature : forest.FloatFeatures) {
        flatFeatureIndices.push_back(floatFeature.FlatFeatureIndex);
    }

    for (const TOneHotFeature& oneHotFeature: forest.OneHotFeatures) {
        flatFeatureIndices.push_back(oneHotFeature.CatFeatureIndex);
    }

    for (const TCtrFeature& ctrFeature : forest.CtrFeatures) {
        const TFeatureCombination& combination = ctrFeature.Ctr.Base.Projection;
        flatFeatureIndices.push_back(forest.CatFeatures[combination.CatFeatures[0]].FlatFeatureIndex);
    }

    return flatFeatureIndices;
}

static bool HasComplexCtrs(const TObliviousTrees& forest) {
    for (const TCtrFeature& ctrFeature : forest.CtrFeatures) {
        const TFeatureCombination& combination = ctrFeature.Ctr.Base.Projection;
        if (!combination.IsSingleCatFeature()) {
            return true;
        }
    }
    return false;
}

static TVector<TVector<ui8>> TransposeBinarizedFeatures(const TVector<ui8>& allBinarizedFeatures, size_t documentCount) {
    CB_ENSURE(documentCount > 0, "Document block must be non-empty.");
    const size_t featuresCount = allBinarizedFeatures.size() / documentCount;

    TVector<TVector<ui8>> binarizedFeaturesByDocument(documentCount, TVector<ui8>(featuresCount));

    for (size_t documentIdx = 0; documentIdx < documentCount; ++documentIdx) {
        for (size_t featureIdx = 0; featureIdx < featuresCount; ++featureIdx) {
            binarizedFeaturesByDocument[documentIdx][featureIdx] = allBinarizedFeatures[featureIdx * documentCount + documentIdx];
        }
    }

    return binarizedFeaturesByDocument;
}

static TVector<TVector<double>> CalcShapValuesForDocumentBlock(const TFullModel& model,
                                                               const TPool& pool,
                                                               size_t start,
                                                               size_t end,
                                                               NPar::TLocalExecutor& localExecutor,
                                                               int dimension) {
    CB_ENSURE(!HasComplexCtrs(model.ObliviousTrees), "Model uses complex Ctr features. This is not allowed for SHAP values calculation");

    const TObliviousTrees& forest = model.ObliviousTrees;
    const size_t documentCount = end - start;

    TVector<ui8> allBinarizedFeatures = BinarizeFeatures(model, pool, start, end);
    TVector<TVector<ui8>> binarizedFeaturesByDocument = TransposeBinarizedFeatures(allBinarizedFeatures, documentCount);
    allBinarizedFeatures.clear();

    const int flatFeatureCount = pool.Docs.GetFactorsCount();
    TVector<int> binFeaturesMapping = MapFeatures(forest);
    TVector<TVector<double>> shapValues(documentCount, TVector<double>(flatFeatureCount + 1, 0.0));

    NPar::TLocalExecutor::TExecRangeParams blockParams(0, documentCount);
    localExecutor.ExecRange([&] (int documentIdx) {
        const size_t treeCount = forest.GetTreeCount();
        for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
            TVector<TVector<size_t>> subtreeSizes = CalcSubtreeSizesForTree(forest, treeIdx);
            TVector<TFeaturePathElement> initialFeaturePath;
            CalcShapValuesRecursive(forest, binFeaturesMapping, binarizedFeaturesByDocument[documentIdx], treeIdx, /*depth*/ 0, subtreeSizes, dimension,
                                    /*nodeIdx*/ 0, initialFeaturePath, /*zeroPathFraction*/ 1, /*onePathFraction*/ 1, /*feature*/ -1,
                                    &shapValues[documentIdx]);

            shapValues[documentIdx][flatFeatureCount] += CalcMeanValueForTree(forest, subtreeSizes, treeIdx, dimension);
        }
    }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);

    return shapValues;
}

TVector<TVector<double>> CalcShapValues(const TFullModel& model,
                                        const TPool& pool,
                                        int threadCount,
                                        int dimension) {
    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);
    TVector<TVector<double>> result = CalcShapValuesForDocumentBlock(model, pool, /*start*/ 0, pool.Docs.GetDocCount(), localExecutor,  dimension);
    return result;
}

static void OutputShapValues(const TVector<TVector<double>>& shapValues,
                             TFileOutput& out) {
    for (size_t documentIdx = 0; documentIdx < shapValues.size(); ++documentIdx) {
        int featureCount = shapValues[documentIdx].size();
        for (int featureIdx = 0; featureIdx < featureCount; ++featureIdx) {
            out << shapValues[documentIdx][featureIdx] << (featureIdx + 1 == featureCount ? '\n' : '\t');
        }
    }
}

void CalcAndOutputShapValues(const TFullModel& model,
                             const TPool& pool,
                             const TString& outputPath,
                             int threadCount,
                             int dimension) {
    const size_t documentCount = pool.Docs.GetDocCount();

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);

    TFileOutput out(outputPath);

    const size_t documentBlockSize = CB_THREAD_LIMIT; // least necessary for threading

    for (size_t start = 0; start < documentCount; start += documentBlockSize) {
        size_t end = Min(start + documentBlockSize, pool.Docs.GetDocCount());
        TVector<TVector<double>> shapValues = CalcShapValuesForDocumentBlock(model, pool, start, end, localExecutor, dimension);
        OutputShapValues(shapValues, out);
    }
}
