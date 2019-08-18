#include "feature_grouping.h"

#include "quantized_features_info.h"

#include <catboost/libs/helpers/exception.h>


namespace NCB {

    TVector<TFeaturesGroup> CreateFeatureGroups(
        const TFeaturesLayout& featuresLayout,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TVector<TMaybe<TExclusiveBundleIndex>>& flatFeatureIndexToBundlePart,
        const TVector<TMaybe<TPackedBinaryIndex>>& flatFeatureIndexToPackedBinaryIndex,
        const TFeaturesGroupingOptions& options)
    {
        CB_ENSURE(
            options.MaxFeaturesPerBundle == 1 ||
            options.MaxFeaturesPerBundle == 2 ||
            options.MaxFeaturesPerBundle == 4,
            "Currently it is possible to group only 1, 2 or 4 features");
        TVector<TFeaturesGroup> groups;
        TFeaturesGroup group;
        featuresLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
            [&] (const TFloatFeatureIdx& floatFeatureIdx) {
                /* Currently grouped only float features with number of bins up to 256 */
                if (quantizedFeaturesInfo.GetBinCount(floatFeatureIdx) > 256)
                    return;

                const ui32 externalIdx = featuresLayout.GetExternalFeatureIdx(*floatFeatureIdx, EFeatureType::Float);
                const bool isInExclusiveBundle = flatFeatureIndexToBundlePart[externalIdx].Defined();
                const bool isPackedBinary = flatFeatureIndexToPackedBinaryIndex[externalIdx].Defined();
                if (isInExclusiveBundle || isPackedBinary)
                    return;

                group.Add(TFeaturesGroupPart{EFeatureType::Float, *floatFeatureIdx});

                if (group.Parts.size() == options.MaxFeaturesPerBundle) {
                    groups.emplace_back(group);
                    group = {};
                }
            });

        if (!group.Parts.empty()) {
            if (group.Parts.size() == 3) {
                // split to 2 groups with 1 and 2 parts to fit later in ui8 and ui16
                TFeaturesGroup firstGroup, secondGroup;
                firstGroup.Add(group.Parts[0]);
                firstGroup.Add(group.Parts[1]);
                secondGroup.Add(group.Parts[2]);
                groups.emplace_back(firstGroup);
                groups.emplace_back(secondGroup);
            } else {
                groups.emplace_back(group);
            }
        }
        return groups;
    }
}
