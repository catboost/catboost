#include "score_calcers.h"

#include "split.h"

#include <util/generic/bitops.h>
#include <util/generic/cast.h>


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
    }
}
