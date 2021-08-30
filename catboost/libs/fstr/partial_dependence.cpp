#include "partial_dependence.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/algo/apply.h>

#include <util/generic/utility.h>


using namespace NCB;

namespace {
    struct TFloatFeatureBucketRange {
        int FeatureIdx = -1;
        int Start = 0;
        int End = -1;
        int NumOfBuckets = 0;

        TFloatFeatureBucketRange() = default;

        TFloatFeatureBucketRange(int featureIdx, int numOfBorders)
                : FeatureIdx(featureIdx)
                , Start(0)
                , End(numOfBorders + 1)
                , NumOfBuckets(numOfBorders + 1)
        {
        }

        void Update(int borderIdx, bool isGreater) {
            if (isGreater && borderIdx >= Start) {
                Start = borderIdx + 1;
            } else if (!isGreater && borderIdx < End) {
                End = borderIdx + 1;
            }
        }
    };
} //anonymous

TVector<TFloatFeatureBucketRange> PrepareFeatureRanges(
        const TFullModel& model,
        const TVector<int>& featuresIdx
) {
    TVector<TFloatFeatureBucketRange> featureRanges(0);
    if (featuresIdx.size() < 2) {
        featureRanges.push_back(TFloatFeatureBucketRange(-1, 0));
    }
    for (int idx : featuresIdx) {
        const auto& feature = model.ModelTrees->GetFloatFeatures()[idx];
        const auto& range = TFloatFeatureBucketRange(feature.Position.Index, feature.Borders.size());
        featureRanges.push_back(range);
    }
    return featureRanges;
}

TVector<TVector<TFloatFeatureBucketRange>> CalculateBucketRangesAndWeightsOblivious(
        const TFullModel& model,
        const TVector<int>& features,
        const TVector<ui32>& borderIdxForSplit,
        const TVector<double>& leafWeights,
        TVector<double>* leafWeightsNew,
        NPar::ILocalExecutor* localExecutor
) {
    CB_ENSURE_INTERNAL(model.IsOblivious(), "Partial dependence is supported only for symmetric trees");

    const auto& binSplits = model.ModelTrees->GetBinFeatures();
    const auto& treeSplitOffsets = model.ModelTrees->GetModelTreeData()->GetTreeStartOffsets();
    auto applyData = model.ModelTrees->GetApplyData();
    const auto& leafOffsets = applyData->TreeFirstLeafOffsets;
    const auto& treeSizes = model.ModelTrees->GetModelTreeData()->GetTreeSizes();
    const auto& treeSplits = model.ModelTrees->GetModelTreeData()->GetTreeSplits();
    size_t leafNum = model.ModelTrees->GetModelTreeData()->GetLeafValues().size();

    const TVector<TFloatFeatureBucketRange> defaultRanges = PrepareFeatureRanges(model, features);
    TVector<TVector<TFloatFeatureBucketRange>> leafBucketRanges(leafNum, defaultRanges);

    int treeCount = model.ModelTrees->GetTreeCount();
    NPar::ILocalExecutor::TExecRangeParams blockParams(0, treeCount);
    localExecutor->ExecRange([&] (size_t treeId) {
        size_t offset = leafOffsets[treeId];
        size_t treeDepth = treeSizes[treeId];
        TVector<int> depthsToExplore;
        TVector<int> splitsToExplore;
        for (size_t depthIdx = 0; depthIdx < treeDepth; ++depthIdx) {
            int splitIdx = treeSplits[treeSplitOffsets[treeId] + depthIdx];
            const auto& split = binSplits[splitIdx];
            if (std::find(features.begin(), features.end(), split.FloatFeature.FloatFeature) != features.end()) {
                depthsToExplore.push_back(depthIdx);
                splitsToExplore.push_back(splitIdx);
            }
        }

        for (size_t leafIdx = 0; leafIdx < 1 << treeDepth; ++leafIdx) {
            for (int mask = 0; mask < 1 << depthsToExplore.size(); ++mask) {
                TVector<TFloatFeatureBucketRange> featureRanges = defaultRanges;

                int newLeafIdx = leafIdx;
                for (size_t splitIdx = 0; splitIdx < splitsToExplore.size(); ++splitIdx) {
                    int depth = depthsToExplore[splitIdx];
                    int decision = (mask >> splitIdx) & 1;
                    newLeafIdx = (newLeafIdx & ~(1UL << depth)) | (decision << depth);
                    const auto& split = binSplits[splitsToExplore[splitIdx]];
                    for (auto& range: featureRanges) {
                        if (range.FeatureIdx == split.FloatFeature.FloatFeature) {
                            int borderIdx = borderIdxForSplit[splitsToExplore[splitIdx]];
                            range.Update(borderIdx, decision);
                        }
                    }
                }
                leafBucketRanges[offset + newLeafIdx] = featureRanges;
                (*leafWeightsNew)[offset + leafIdx] += leafWeights[offset + newLeafIdx];
            }
        }
    }, blockParams, NPar::TLocalExecutor::WAIT_COMPLETE);

    return leafBucketRanges;
}

TVector<double> MergeBucketRanges(
        const TFullModel& model,
        const TVector<int>& features,
        const TDataProvider& dataProvider,
        const TVector<TVector<TFloatFeatureBucketRange>>& leafBucketRanges,
        const TVector<double> leafWeights
) {
    const auto& leafValues = model.ModelTrees->GetModelTreeData()->GetLeafValues();
    TVector<TFloatFeatureBucketRange> defaultRanges = PrepareFeatureRanges(model, features);
    CB_ENSURE(defaultRanges.size() == 2, "Number of features must be 2");

    int columnNum = defaultRanges[1].NumOfBuckets;
    int rowNum = defaultRanges[0].NumOfBuckets;
    int numOfBucketsTotal = rowNum * columnNum;
    TVector<double> edges(numOfBucketsTotal);

    for (size_t leafIdx = 0; leafIdx < leafValues.size(); ++leafIdx) {
        const auto& ranges = leafBucketRanges[leafIdx];
        double leafValue = leafValues[leafIdx];
        for (int rowIdx = ranges[0].Start; rowIdx < ranges[0].End; ++rowIdx) {
            if (ranges[1].Start < ranges[1].End) {
                edges[rowIdx * columnNum + ranges[1].Start] += leafValue * leafWeights[leafIdx];
                if (ranges[1].End != columnNum) {
                    edges[rowIdx * columnNum + ranges[1].End] -= leafValue * leafWeights[leafIdx];
                } else if (rowIdx < rowNum - 1) {
                    edges[(rowIdx + 1) * columnNum] -= leafValue * leafWeights[leafIdx];
                }
            }
        }
    }

    TVector<double> predictionsByBuckets(numOfBucketsTotal);
    double acc = 0;
    for (int idx = 0; idx < numOfBucketsTotal; ++idx) {
        acc += edges[idx];
        predictionsByBuckets[idx] = acc;
    }

    size_t numOfDocuments = dataProvider.GetObjectCount();
    for (size_t idx = 0; idx < predictionsByBuckets.size(); ++idx) {
        predictionsByBuckets[idx] /= numOfDocuments;
    }
    return predictionsByBuckets;
}

TVector<double> CalculatePartialDependence(
        const TFullModel& model,
        const TVector<int>& features,
        const TDataProvider& dataProvider,
        const TVector<ui32>& borderIdxForSplit,
        const TVector<double> leafWeights,
        NPar::ILocalExecutor* localExecutor
) {
    TVector<double> leafWeightsNew(leafWeights.size(), 0.0);
    const auto& leafBucketRanges = CalculateBucketRangesAndWeightsOblivious(
            model,
            features,
            borderIdxForSplit,
            leafWeights,
            &leafWeightsNew,
            localExecutor
    );

    TVector<double> predictionsByBuckets = MergeBucketRanges(model, features, dataProvider, leafBucketRanges, leafWeightsNew);

    return predictionsByBuckets;
}

TVector<double> GetPartialDependence(
        const TFullModel& model,
        const TVector<int>& features,
        const NCB::TDataProviderPtr dataProvider,
        int threadCount
) {
    CB_ENSURE(model.ModelTrees->GetDimensionsCount() == 1,  "Is not supported for multiclass");
    CB_ENSURE(model.GetNumCatFeatures() == 0, "Models with categorical features are not supported");
    CB_ENSURE(features.size() > 0 && features.size() <= 2, "Number of features should be equal to one or two");
    //TODO(eermishkina): support non symmetric trees
    CB_ENSURE(model.IsOblivious(), "Partial dependence is supported only for symmetric trees");

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);

    TVector<double> leafWeights = CollectLeavesStatistics(*dataProvider, model, &localExecutor);

    const auto& binSplits = model.ModelTrees->GetBinFeatures();

    TVector<ui32> borderIdxForSplit(binSplits.size(), std::numeric_limits<ui32>::infinity());
    ui32 splitIdx = 0;
    for (const auto& feature : model.ModelTrees->GetFloatFeatures()) {
        if (splitIdx == binSplits.size() ||
            binSplits[splitIdx].Type != ESplitType::FloatFeature ||
            binSplits[splitIdx].FloatFeature.FloatFeature > feature.Position.Index
                ) {
            continue;
        }
        CB_ENSURE_INTERNAL(binSplits[splitIdx].FloatFeature.FloatFeature >= feature.Position.Index, "Only float features are supported");
        for (ui32 idx = 0; idx < feature.Borders.size() && binSplits[splitIdx].FloatFeature.FloatFeature == feature.Position.Index; ++idx) {
            if (abs(binSplits[splitIdx].FloatFeature.Split - feature.Borders[idx]) < 1e-15) {
                borderIdxForSplit[splitIdx] = idx;
                ++splitIdx;
            }
        }
    }

    TVector<double> predictionsByBuckets = CalculatePartialDependence(
            model,
            features,
            *dataProvider,
            borderIdxForSplit,
            leafWeights,
            &localExecutor
    );
    return predictionsByBuckets;
}
