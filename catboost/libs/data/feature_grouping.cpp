#include "feature_grouping.h"

#include "quantized_features_info.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>


namespace NCB {

    TVector<TFeaturesGroup> CreateFeatureGroups(
        const TFeaturesLayout& featuresLayout,
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TVector<TMaybe<TExclusiveBundleIndex>>& flatFeatureIndexToBundlePart,
        const TVector<TMaybe<TPackedBinaryIndex>>& flatFeatureIndexToPackedBinaryIndex,
        const TFeaturesGroupingOptions& options)
    {
        CB_ENSURE(
            options.MaxFeaturesPerBundle == 2 || options.MaxFeaturesPerBundle == 4,
            "Currently it is possible to group only 2 or 4 features");
        TVector<TFeaturesGroup> groups;
        TFeaturesGroup group;
        featuresLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
            [&] (const TFloatFeatureIdx& floatFeatureIdx) {
                /* Currently grouped only float features with number of bins up to 256 */
                ui32 bucketCount = quantizedFeaturesInfo.GetBinCount(floatFeatureIdx);
                if (bucketCount > 256)
                    return;

                const ui32 externalIdx = featuresLayout.GetExternalFeatureIdx(*floatFeatureIdx, EFeatureType::Float);
                const bool isInExclusiveBundle = flatFeatureIndexToBundlePart[externalIdx].Defined();
                const bool isPackedBinary = flatFeatureIndexToPackedBinaryIndex[externalIdx].Defined();
                if (isInExclusiveBundle || isPackedBinary)
                    return;

                group.Add(TFeaturesGroupPart{EFeatureType::Float, *floatFeatureIdx, bucketCount});

                if (group.Parts.size() == options.MaxFeaturesPerBundle) {
                    groups.emplace_back(group);
                    group = {};
                }
            });

        // it is unreasonable to group 1 feature or 3 features
        if (group.Parts.size() == 1 || group.Parts.size() == 3) {
            group.PopLastFeature();
        }
        if (!group.Parts.empty()) {
            groups.emplace_back(group);
        }
        for (auto groupIdx : xrange(groups.size())) {
            CATBOOST_DEBUG_LOG << "Group #" << groupIdx << ":";
            for (auto& part : groups[groupIdx].Parts) {
                CATBOOST_DEBUG_LOG << " " << part.FeatureIdx;
            }
            CATBOOST_DEBUG_LOG << Endl;
        }
        return groups;
    }
}
