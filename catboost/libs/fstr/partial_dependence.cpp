#include "partial_dependence.h"

#include <catboost/private/libs/algo/apply.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/data/data_provider.h>
#include <util/generic/utility.h>


using namespace NCB;

namespace {
    struct TFloatFeatureBucketRange {
        int featureIdx = -1;
        int start = 0;
        int end = 1;
        int numOfBuckets = 1;

        TFloatFeatureBucketRange() = default;

        TFloatFeatureBucketRange(int featureIdx, int numOfBorders)
                : featureIdx(featureIdx)
                , start(0)
                , end(numOfBorders + 1)
                , numOfBuckets(numOfBorders + 1)
        {
        }

        void update(int borderIdx, bool isGreater) {
            if (isGreater && borderIdx >= start) {
                start = borderIdx + 1;
            } else if (!isGreater && borderIdx < end) {
                end = borderIdx + 1;
            }
        }
    };
} //anonymous

TVector<TFloatFeatureBucketRange> prepareFeatureRanges(
        const TFullModel& model,
        const TVector<int>& featuresIdx
) {
    TVector<TFloatFeatureBucketRange> featureRanges(0);
    if (featuresIdx.size() < 2) {
        featureRanges.push_back(TFloatFeatureBucketRange());
    }
    for (const auto& idx: featuresIdx) {
        const auto feature = model.ModelTrees->GetFloatFeatures()[idx];
        const auto featurePosition = feature.Position.Index;
        const auto borders = feature.Borders.size();
        const auto range = TFloatFeatureBucketRange(featurePosition, borders);
        featureRanges.push_back(range);
    }
    return featureRanges;
}

void UpdateFeatureRanges(
        const TFullModel& model,
        TVector<TFloatFeatureBucketRange>* featureRanges,
        const TVector<int>& splits,
        const TVector<ui32>& borderIdxForSplit,
        int mask
) {
    const auto& binSplits = model.ModelTrees->GetBinFeatures();

    for (size_t splitIdx = 0; splitIdx < splits.size(); ++splitIdx) {
        bool decision = (mask >> splitIdx) & 1;
        const auto split = binSplits[splits[splitIdx]];
        for (auto& range: *featureRanges) {
            if (range.featureIdx == split.FloatFeature.FloatFeature) {
                int borderIdx = borderIdxForSplit[splits[splitIdx]];
                range.update(borderIdx, decision);
            }
        }
    }

}

void UpdatePartialDependence(
        TVector<double>* predictionsByBuckets,
        const TVector<TFloatFeatureBucketRange>& featureRanges,
        double prediction
) {
    CB_ENSURE(featureRanges.size() == 2,  "Wrong number of features");
    int columnNum = featureRanges[1].numOfBuckets;
    for (int rowIdx = featureRanges[0].start; rowIdx < featureRanges[0].end; ++rowIdx) {
        for (int columnIdx = featureRanges[1].start; columnIdx < featureRanges[1].end; ++columnIdx) {
            (*predictionsByBuckets)[rowIdx * columnNum + columnIdx] += prediction;
        }
    }
}

void CalculatePartialDependenceOblivious(
        const TFullModel& model,
        const TVector<int>& features,
        const TDataProvider& dataProvider,
        const TVector<ui32>& borderIdxForSplit,
        const TVector<ui32> treeLeafIdxes,
        TVector<double>* predictionsByBuckets
) {
    const auto& binSplits = model.ModelTrees->GetBinFeatures();
    const auto& treeSplitOffsets = model.ModelTrees->GetTreeStartOffsets();

    for (ui32 docIdx = 0; docIdx < dataProvider.GetObjectCount(); ++docIdx) {
        size_t offset = 0;
        for (size_t treeId = 0; treeId < model.ModelTrees->GetTreeCount(); ++treeId) {
            size_t treeDepth = model.ModelTrees->GetTreeSizes().at(treeId);
            int leafIdx = offset + treeLeafIdxes[treeId + docIdx * model.GetTreeCount()];
            TVector<int> depthsToExplore;
            TVector<int> splitsToExplore;
            for (size_t depthIdx = 0; depthIdx < treeDepth; ++depthIdx) {
                int splitIdx = model.ModelTrees->GetTreeSplits()[treeSplitOffsets[treeId] + depthIdx];
                const auto split = binSplits[splitIdx];
                if (std::find(features.begin(), features.end(), split.FloatFeature.FloatFeature) != features.end()) {
                    depthsToExplore.push_back(depthIdx);
                    splitsToExplore.push_back(splitIdx);
                }
            }

            int numberOfMasks = 1 << depthsToExplore.size();
            for (int mask = 0; mask < numberOfMasks; ++mask) {
                TVector<TFloatFeatureBucketRange> featureRanges = prepareFeatureRanges(model, features);

                int newLeafIdx = leafIdx;
                for (size_t splitIdx = 0; splitIdx < splitsToExplore.size(); ++splitIdx) {
                    int depth = depthsToExplore[splitIdx];
                    int decision = (mask >> splitIdx) & 1;
                    newLeafIdx = (newLeafIdx & ~(1UL << depth)) | (decision << depth);
                }
                UpdateFeatureRanges(model, &featureRanges, splitsToExplore, borderIdxForSplit, mask);
                double newLeafValue = model.ModelTrees->GetLeafValues()[newLeafIdx];
                UpdatePartialDependence(predictionsByBuckets, featureRanges, newLeafValue);
            }
            offset += (1ull << treeDepth);
        }
    }
    size_t numOfDocuments = dataProvider.GetObjectCount();
    for (size_t idx = 0; idx < predictionsByBuckets->size(); ++idx) {
        (*predictionsByBuckets)[idx] /= numOfDocuments;
    }
}


TVector<double> CalculatePartialDependence(
        const TFullModel& model,
        const TVector<int>& features,
        const TDataProvider& dataProvider,
        const TVector<ui32>& borderIdxForSplit,
        const TVector<ui32> treeLeafIdxes
) {
    size_t numOfBuckets = 1;
    for (const auto& featureIdx: features) {
        const auto feature = model.ModelTrees->GetFloatFeatures()[featureIdx];
        numOfBuckets *= feature.Borders.size() + 1;
    }
    TVector<double> predictionsByBuckets(numOfBuckets);

    CalculatePartialDependenceOblivious(
            model,
            features,
            dataProvider,
            borderIdxForSplit,
            treeLeafIdxes,
            &predictionsByBuckets
    );

    return predictionsByBuckets;
}

TVector<double> GetPartialDependence(
        const TFullModel& model,
        const TVector<int>& features,
        const NCB::TDataProviderPtr dataProvider
) {
    CB_ENSURE(model.ModelTrees->GetDimensionsCount() == 1,  "Is not supported for multiclass");
    CB_ENSURE(model.GetNumCatFeatures() == 0, "Model with categorical features are not supported");
    CB_ENSURE(features.size() > 0 && features.size() <= 2, "Number of features should be equal to one or two");

    TVector<ui32> leafIdxes = CalcLeafIndexesMulti(model, (*dataProvider).ObjectsData, 0, 0);

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
        Y_ASSERT(binSplits[splitIdx].FloatFeature.FloatFeature >= feature.Position.Index);
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
            leafIdxes
    );
    return predictionsByBuckets;
}

