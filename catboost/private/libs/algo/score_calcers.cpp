#include "score_calcers.h"

#include "split.h"

#include <catboost/libs/helpers/short_vector_ops.h>

#include <util/generic/bitops.h>
#include <util/generic/cast.h>

void TCosineScoreCalcer::AddLeafPlain(int splitIdx, const TBucketStats& leftStats, const TBucketStats& rightStats) {
    NSimdOps::UpdateScoreBinKernelPlain(L2Regularizer, &rightStats.SumWeightedDelta, &leftStats.SumWeightedDelta, &Scores[splitIdx][0]);
}


void TCosineScoreCalcer::AddLeafOrdered(int splitIdx, const TBucketStats& leftStats, const TBucketStats& rightStats) {
    NSimdOps::UpdateScoreBinKernelOrdered(L2Regularizer, &rightStats.SumWeightedDelta, &leftStats.SumWeightedDelta, &Scores[splitIdx][0]);
}


void TL2ScoreCalcer::AddLeafPlain(int splitIdx, const TBucketStats& leftStats, const TBucketStats& rightStats) {
    const double rightAvrg = CalcAverage(
        rightStats.SumWeightedDelta,
        rightStats.SumWeight,
        L2Regularizer
    );
    const double leftAvrg = CalcAverage(
        leftStats.SumWeightedDelta,
        leftStats.SumWeight,
        L2Regularizer
    );
    AddLeaf(splitIdx, rightAvrg, rightStats);
    AddLeaf(splitIdx, leftAvrg, leftStats);
}


void TL2ScoreCalcer::AddLeafOrdered(int splitIdx, const TBucketStats& leftStats, const TBucketStats& rightStats) {
    const double rightAvrg = CalcAverage(
        rightStats.SumDelta,
        rightStats.Count,
        L2Regularizer
    );
    const double leftAvrg = CalcAverage(
        leftStats.SumDelta,
        leftStats.Count,
        L2Regularizer
    );
    AddLeaf(splitIdx, rightAvrg, rightStats);
    AddLeaf(splitIdx, leftAvrg, leftStats);
}


int CalcSplitsCount(
    const TSplitEnsembleSpec& splitEnsembleSpec,
    int bucketCount,
    ui32 oneHotMaxSize
) {
    switch (splitEnsembleSpec.Type) {
        case ESplitEnsembleType::OneFeature:
            return (splitEnsembleSpec.OneSplitType == ESplitType::OneHotFeature) ?
                bucketCount :
                bucketCount - 1;
        case ESplitEnsembleType::BinarySplits:
            return (int)GetValueBitCount(bucketCount - 1);
        case ESplitEnsembleType::ExclusiveBundle:
            {
                size_t binCount = 0;
                for (const auto& bundlePart : splitEnsembleSpec.ExclusiveFeaturesBundle.Parts) {
                    if (bundlePart.FeatureType == EFeatureType::Categorical) {
                        if ((bundlePart.Bounds.GetSize() + 1) <= oneHotMaxSize) {
                            binCount += bundlePart.Bounds.GetSize() + 1;
                        }
                    } else {
                        binCount += bundlePart.Bounds.GetSize();
                    }
                }
                return SafeIntegerCast<int>(binCount);
            }
        case ESplitEnsembleType::FeaturesGroup:
            {
                const auto& featuresGroup = splitEnsembleSpec.FeaturesGroup;
                // for each part: part.BucketCount - 1
                return featuresGroup.TotalBucketCount - featuresGroup.Parts.size();
            }
        default:
            CB_ENSURE(false, "Unexpected split ensemble type");
    }
    Y_UNREACHABLE();
}
