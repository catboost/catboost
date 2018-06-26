#include "shap_values.h"
#include "util.h"

#include <catboost/libs/algo/index_calcer.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/profile_info.h>

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

    struct TShapValue {
        int Feature;
        double Value;

        TShapValue() = default;

        TShapValue(int feature, double value)
            : Feature(feature)
            , Value(value)
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

static size_t CalcLeafToFallForDocument(const TObliviousTrees& forest, size_t treeIdx, const TVector<ui8>& binarizedFeatures) {
    size_t leafIdx = 0;
    for (int depth = 0; depth < forest.TreeSizes[treeIdx]; ++depth) {
        const TRepackedBin& split = forest.GetRepackedBins()[forest.TreeStartOffsets[treeIdx] + depth];
        leafIdx |= ((binarizedFeatures[split.FeatureIndex] ^ split.XorMask) >= split.SplitIdx) << depth;
    }
    return leafIdx;
}

static void CalcShapValuesForLeafRecursive(
    const TObliviousTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    size_t documentLeafIdx,
    size_t treeIdx,
    int depth,
    const TVector<TVector<double>>& subtreeWeights,
    int dimension,
    size_t nodeIdx,
    const TVector<TFeaturePathElement>& oldFeaturePath,
    double zeroPathsFraction,
    double onePathsFraction,
    int feature,
    TVector<TShapValue>* shapValues
) {
    TVector<TFeaturePathElement> featurePath = ExtendFeaturePath(oldFeaturePath, zeroPathsFraction, onePathsFraction, feature);
    auto firstLeafPtr = forest.GetFirstLeafPtrForTree(treeIdx);
    if (depth == forest.TreeSizes[treeIdx]) {
        for (size_t elementIdx = 1; elementIdx < featurePath.size(); ++elementIdx) {
            TVector<TFeaturePathElement> unwoundPath = UnwindFeaturePath(featurePath, elementIdx);
            double weightSum = 0.0;
            for (const TFeaturePathElement& unwoundPathElement : unwoundPath) {
                weightSum += unwoundPathElement.Weight;
            }
            const TFeaturePathElement& element = featurePath[elementIdx];
            const int approxDimension = forest.ApproxDimension;

            const TVector<int>& flatFeatures = combinationClassFeatures[element.Feature];

            for (int flatFeatureIdx : flatFeatures) {
                double addValue = weightSum * (element.OnePathsFraction - element.ZeroPathsFraction)
                                * firstLeafPtr[nodeIdx * approxDimension + dimension] / flatFeatures.size();
                const auto sameFeatureShapValue = FindIf(
                    shapValues->begin(),
                    shapValues->end(),
                    [flatFeatureIdx](const TShapValue& shapValue) {
                        return shapValue.Feature == flatFeatureIdx;
                    }
                );
                if (sameFeatureShapValue == shapValues->end()) {
                    shapValues->emplace_back(flatFeatureIdx, addValue);
                } else {
                    sameFeatureShapValue->Value += addValue;
                }
            }
        }
    } else {
        const TRepackedBin& split = forest.GetRepackedBins()[forest.TreeStartOffsets[treeIdx] + depth];
        const int splitFeature = split.FeatureIndex;

        double newZeroPathsFraction = 1.0;
        double newOnePathsFraction = 1.0;

        const int combinationClass = binFeatureCombinationClass[splitFeature];

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

        const size_t goNodeIdx = nodeIdx | (documentLeafIdx & (size_t(1) << depth));
        const size_t skipNodeIdx = goNodeIdx ^ (1 << depth);

        if (!FuzzyEquals(subtreeWeights[depth + 1][goNodeIdx], 0.0)) {
            double newZeroPathsFractionGoNode = newZeroPathsFraction * subtreeWeights[depth + 1][goNodeIdx] / subtreeWeights[depth][nodeIdx];
            CalcShapValuesForLeafRecursive(
                forest,
                binFeatureCombinationClass,
                combinationClassFeatures,
                documentLeafIdx,
                treeIdx,
                depth + 1,
                subtreeWeights,
                dimension,
                goNodeIdx,
                featurePath,
                newZeroPathsFractionGoNode,
                newOnePathsFraction,
                combinationClass,
                shapValues);
        }

        if (!FuzzyEquals(subtreeWeights[depth + 1][skipNodeIdx], 0.0)) {
            double newZeroPathsFractionSkipNode = newZeroPathsFraction * subtreeWeights[depth + 1][skipNodeIdx] / subtreeWeights[depth][nodeIdx];
            CalcShapValuesForLeafRecursive(
                forest,
                binFeatureCombinationClass,
                combinationClassFeatures,
                documentLeafIdx,
                treeIdx,
                depth + 1,
                subtreeWeights,
                dimension,
                skipNodeIdx,
                featurePath,
                newZeroPathsFractionSkipNode,
                /*onePathFraction*/ 0,
                combinationClass,
                shapValues);
        }
    }
}

static inline void CalcShapValuesForLeaf(
    const TObliviousTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    const TVector<TVector<int>>& combinationClassFeatures,
    size_t documentLeafIdx,
    size_t treeIdx,
    const TVector<TVector<double>>& subtreeWeights,
    int dimension,
    TVector<TShapValue>* shapValues
) {
    shapValues->clear();

    TVector<TFeaturePathElement> initialFeaturePath;
    CalcShapValuesForLeafRecursive(
        forest,
        binFeatureCombinationClass,
        combinationClassFeatures,
        documentLeafIdx,
        treeIdx,
        /*depth*/ 0,
        subtreeWeights,
        dimension,
        /*nodeIdx*/ 0,
        initialFeaturePath,
        /*zeroPathFraction*/ 1,
        /*onePathFraction*/ 1,
        /*feature*/ -1,
        shapValues);
}

static double CalcMeanValueForTree(
    const TObliviousTrees& forest,
    const TVector<TVector<double>>& subtreeWeights,
    size_t treeIdx,
    int dimension
) {
    double meanValue = 0.0;
    auto firstLeafPtr = forest.GetFirstLeafPtrForTree(treeIdx);
    const size_t maxDepth = forest.TreeSizes[treeIdx];

    for (size_t leafIdx = 0; leafIdx < (size_t(1) << maxDepth); ++leafIdx) {
        const int approxDimension = forest.ApproxDimension;

        meanValue += firstLeafPtr[leafIdx * approxDimension + dimension] * subtreeWeights[maxDepth][leafIdx];
    }

    meanValue /= subtreeWeights[0][0];

    return meanValue;
}

static TVector<TVector<double>> CalcSubtreeWeightsForTree(const TVector<double>& leafWeights, int treeDepth) {
    TVector<TVector<double>> subtreeWeights(treeDepth + 1);

    subtreeWeights[treeDepth] = leafWeights;

    for (int depth = treeDepth - 1; depth >= 0; --depth) {
        const size_t nodeCount = size_t(1) << depth;
        subtreeWeights[depth].resize(nodeCount);
        for (size_t nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx) {
            subtreeWeights[depth][nodeIdx] = subtreeWeights[depth + 1][nodeIdx] + subtreeWeights[depth + 1][nodeIdx ^ (1 << depth)];
        }
    }
    return subtreeWeights;
}

static void MapBinFeaturesToClasses(
    const TObliviousTrees& forest,
    TVector<int>* binFeatureCombinationClass,
    TVector<TVector<int>>* combinationClassFeatures
) {
    const int featureCount = forest.GetEffectiveBinaryFeaturesBucketsCount();

    TVector<TVector<int>> binFeaturesCombinations;
    binFeaturesCombinations.reserve(featureCount);

    for (const TFloatFeature& floatFeature : forest.FloatFeatures) {
        binFeaturesCombinations.emplace_back(1, floatFeature.FlatFeatureIndex);
    }

    for (const TOneHotFeature& oneHotFeature: forest.OneHotFeatures) {
        binFeaturesCombinations.emplace_back(1, oneHotFeature.CatFeatureIndex);
    }

    for (const TCtrFeature& ctrFeature : forest.CtrFeatures) {
        const TFeatureCombination& combination = ctrFeature.Ctr.Base.Projection;
        binFeaturesCombinations.emplace_back();
        for (int catFeatureIdx : combination.CatFeatures) {
            binFeaturesCombinations.back().push_back(forest.CatFeatures[catFeatureIdx].FlatFeatureIndex);
        }
    }

    TVector<int> sortedBinFeatures(featureCount);
    Iota(sortedBinFeatures.begin(), sortedBinFeatures.end(), 0);
    Sort(
        sortedBinFeatures.begin(),
        sortedBinFeatures.end(),
        [binFeaturesCombinations](int feature1, int feature2) {
            return binFeaturesCombinations[feature1] < binFeaturesCombinations[feature2];
        }
    );

    *binFeatureCombinationClass = TVector<int>(featureCount);
    *combinationClassFeatures = TVector<TVector<int>>();

    int equivalenceClassesCount = 0;
    for (int featureIdx = 0; featureIdx < featureCount; ++featureIdx) {
        int currentFeature = sortedBinFeatures[featureIdx];
        int previousFeature = featureIdx == 0 ? -1 : sortedBinFeatures[featureIdx - 1];
        if (featureIdx == 0 || binFeaturesCombinations[currentFeature] != binFeaturesCombinations[previousFeature]) {
            combinationClassFeatures->push_back(binFeaturesCombinations[currentFeature]);
            ++equivalenceClassesCount;
        }
        (*binFeatureCombinationClass)[currentFeature] = equivalenceClassesCount - 1;
    }
}

static TVector<ui8> GetBinarizedFeaturesForDocument(const TVector<ui8>& allBinarizedFeatures, size_t documentCount, size_t documentIdx) {
    CB_ENSURE(documentCount > 0, "Document block must be non-empty.");
    const size_t featuresCount = allBinarizedFeatures.size() / documentCount;

    TVector<ui8> binarizedFeaturesForDocument(featuresCount);
    for (size_t featureIdx = 0; featureIdx < featuresCount; ++featureIdx) {
        binarizedFeaturesForDocument[featureIdx] = allBinarizedFeatures[featureIdx * documentCount + documentIdx];
    }

    return binarizedFeaturesForDocument;
}

static void CalcShapValuesForDocumentBlock(
    const TFullModel& model,
    const TPool& pool,
    NPar::TLocalExecutor& localExecutor,
    const TVector<TVector<TVector<TShapValue>>>& shapValuesByLeafForAllTrees,
    const TVector<double>& meanValuesForAllTrees,
    size_t start,
    size_t end,
    TVector<TVector<double>>* shapValuesForAllDocuments
) {
    const TObliviousTrees& forest = model.ObliviousTrees;
    const size_t documentCount = end - start;

    TVector<ui8> binarizedFeaturesForDocumentBlock = BinarizeFeatures(model, pool, start, end);

    const int flatFeatureCount = pool.Docs.GetEffectiveFactorCount();

    const int oldShapValuesSize = shapValuesForAllDocuments->size();
    shapValuesForAllDocuments->resize(oldShapValuesSize + end - start);

    NPar::TLocalExecutor::TExecRangeParams blockParams(0, documentCount);
    localExecutor.ExecRange([&] (size_t documentIdx) {
        TVector<double>& shapValues = (*shapValuesForAllDocuments)[oldShapValuesSize + documentIdx];
        shapValues.assign(flatFeatureCount + 1, 0.0);

        TVector<ui8> binarizedFeatures = GetBinarizedFeaturesForDocument(binarizedFeaturesForDocumentBlock, documentCount, documentIdx);

        const size_t treeCount = forest.GetTreeCount();
        for (size_t treeIdx = 0; treeIdx < treeCount; ++treeIdx) {
            size_t leafIdx = CalcLeafToFallForDocument(forest, treeIdx, binarizedFeatures);
            for (const TShapValue& shapValue : shapValuesByLeafForAllTrees[treeIdx][leafIdx]) {
                shapValues[shapValue.Feature] += shapValue.Value;
            }
            shapValues[flatFeatureCount] += meanValuesForAllTrees[treeIdx];
        }
    }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
}

static void CalcShapValuesByLeafForTreeBlock(
    const TObliviousTrees& forest,
    const TVector<TVector<double>>& leafWeights,
    NPar::TLocalExecutor& localExecutor,
    int dimension,
    int start,
    int end,
    TVector<TVector<TVector<TShapValue>>>* shapValuesByLeafForAllTrees,
    TVector<double>* meanValuesForAllTrees
) {
    TVector<int> binFeatureCombinationClass;
    TVector<TVector<int>> combinationClassFeatures;
    MapBinFeaturesToClasses(forest, &binFeatureCombinationClass, &combinationClassFeatures);

    NPar::TLocalExecutor::TExecRangeParams blockParams(start, end);
    localExecutor.ExecRange([&] (size_t treeIdx) {
        const size_t leafCount = (size_t(1) << forest.TreeSizes[treeIdx]);
        TVector<TVector<TShapValue>>& shapValuesByLeaf = (*shapValuesByLeafForAllTrees)[treeIdx];
        shapValuesByLeaf.resize(leafCount);

        TVector<TVector<double>> subtreeWeights = CalcSubtreeWeightsForTree(leafWeights[treeIdx], forest.TreeSizes[treeIdx]);

        for (size_t leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
            CalcShapValuesForLeaf(
                forest,
                binFeatureCombinationClass,
                combinationClassFeatures,
                leafIdx,
                treeIdx,
                subtreeWeights,
                dimension,
                &shapValuesByLeaf[leafIdx]);

            (*meanValuesForAllTrees)[treeIdx] = CalcMeanValueForTree(forest, subtreeWeights, treeIdx, dimension);
        }
    }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);
}

static void WarnForComplexCtrs(const TObliviousTrees& forest) {
    for (const TCtrFeature& ctrFeature : forest.CtrFeatures) {
        const TFeatureCombination& combination = ctrFeature.Ctr.Base.Projection;
        if (!combination.IsSingleCatFeature()) {
            MATRIXNET_WARNING_LOG << "The model has complex ctrs, so the SHAP values will be calculated approximately." << Endl;
            return;
        }
    }
}

static void PrepareTrees(
    const TFullModel& model,
    const TPool& pool,
    NPar::TLocalExecutor& localExecutor,
    int logPeriod,
    int dimension,
    TVector<TVector<TVector<TShapValue>>>* shapValuesByLeafForAllTrees,
    TVector<double>* meanValuesForAllTrees
) {
    WarnForComplexCtrs(model.ObliviousTrees);

    const size_t treeCount = model.GetTreeCount();
    const size_t treeBlockSize = CB_THREAD_LIMIT; // least necessary for threading

    TFstrLogger treesLogger(treeCount, "trees processed", "Processing trees...", logPeriod);

    // use only if model.ObliviousTrees.LeafWeights is empty
    TVector<TVector<double>> leafWeights;
    if (model.ObliviousTrees.LeafWeights.empty()) {
        leafWeights = CollectLeavesStatistics(pool, model);
    }

    shapValuesByLeafForAllTrees->resize(treeCount);
    meanValuesForAllTrees->resize(treeCount);

    TProfileInfo processTreesProfile(treeCount);

    for (size_t start = 0; start < treeCount; start += treeBlockSize) {
        size_t end = Min(start + treeBlockSize, treeCount);

        processTreesProfile.StartIterationBlock();

        CalcShapValuesByLeafForTreeBlock(
            model.ObliviousTrees,
            model.ObliviousTrees.LeafWeights.empty() ? leafWeights : model.ObliviousTrees.LeafWeights,
            localExecutor,
            dimension,
            start,
            end,
            shapValuesByLeafForAllTrees,
            meanValuesForAllTrees
        );

        processTreesProfile.FinishIterationBlock(end - start);
        auto profileResults = processTreesProfile.GetProfileResults();
        treesLogger.Log(profileResults);
    }
}

TVector<TVector<double>> CalcShapValues(
    const TFullModel& model,
    const TPool& pool,
    int threadCount,
    int logPeriod,
    int dimension
) {
    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);

    TVector<TVector<TVector<TShapValue>>> shapValuesByLeafForAllTrees;
    TVector<double> meanValuesForAllTrees;

    PrepareTrees(
        model,
        pool,
        localExecutor,
        logPeriod,
        dimension,
        &shapValuesByLeafForAllTrees,
        &meanValuesForAllTrees
    );

    const size_t documentCount = pool.Docs.GetDocCount();
    const size_t documentBlockSize = CB_THREAD_LIMIT; // least necessary for threading

    TFstrLogger documentsLogger(documentCount, "documents processed", "Processing documents...", logPeriod);

    TVector<TVector<double>> shapValues;
    shapValues.reserve(documentCount);

    TProfileInfo processDocumentsProfile(documentCount);

    for (size_t start = 0; start < documentCount; start += documentBlockSize) {
        size_t end = Min(start + documentBlockSize, documentCount);

        processDocumentsProfile.StartIterationBlock();

        CalcShapValuesForDocumentBlock(model, pool, localExecutor, shapValuesByLeafForAllTrees, meanValuesForAllTrees, start, end, &shapValues);

        processDocumentsProfile.FinishIterationBlock(end - start);
        auto profileResults = processDocumentsProfile.GetProfileResults();
        documentsLogger.Log(profileResults);
    }

    return shapValues;
}

static void OutputShapValues(const TVector<TVector<double>>& shapValues, TFileOutput& out) {
    for (size_t documentIdx = 0; documentIdx < shapValues.size(); ++documentIdx) {
        int featureCount = shapValues[documentIdx].size();
        for (int featureIdx = 0; featureIdx < featureCount; ++featureIdx) {
            out << shapValues[documentIdx][featureIdx] << (featureIdx + 1 == featureCount ? '\n' : '\t');
        }
    }
}

void CalcAndOutputShapValues(
    const TFullModel& model,
    const TPool& pool,
    const TString& outputPath,
    int threadCount,
    int logPeriod,
    int dimension
) {
    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);

    TVector<TVector<TVector<TShapValue>>> shapValuesByLeafForAllTrees;
    TVector<double> meanValuesForAllTrees;

    PrepareTrees(
        model,
        pool,
        localExecutor,
        logPeriod,
        dimension,
        &shapValuesByLeafForAllTrees,
        &meanValuesForAllTrees
    );

    const size_t documentCount = pool.Docs.GetDocCount();
    const size_t documentBlockSize = CB_THREAD_LIMIT; // least necessary for threading

    TFstrLogger documentsLogger(documentCount, "documents processed", "Processing documents...", logPeriod);

    TProfileInfo processDocumentsProfile(documentCount);

    TFileOutput out(outputPath);
    for (size_t start = 0; start < documentCount; start += documentBlockSize) {
        size_t end = Min(start + documentBlockSize, pool.Docs.GetDocCount());

        processDocumentsProfile.StartIterationBlock();

        TVector<TVector<double>> shapValuesForBlock;
        shapValuesForBlock.reserve(end - start);

        CalcShapValuesForDocumentBlock(model, pool, localExecutor, shapValuesByLeafForAllTrees, meanValuesForAllTrees, start, end, &shapValuesForBlock);
        OutputShapValues(shapValuesForBlock, out);

        processDocumentsProfile.FinishIterationBlock(end - start);
        auto profileResults = processDocumentsProfile.GetProfileResults();
        documentsLogger.Log(profileResults);
    }
}
