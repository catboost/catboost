#include "cat_feature_perfect_hash_helper.h"

#include "util.h"

#include <util/system/guard.h>
#include <util/generic/map.h>

#include <util/generic/ylimits.h>


namespace NCB {

    void TCatFeaturesPerfectHashHelper::UpdatePerfectHashAndMaybeQuantize(
        const TCatFeatureIdx catFeatureIdx,
        TConstMaybeOwningArraySubset<ui32, ui32> hashedCatArraySubset,
        TMaybe<TArrayRef<ui32>*> dstBins
    ) {
        QuantizedFeaturesInfo->CheckCorrectPerTypeFeatureIdx(catFeatureIdx);
        auto& featuresHash = QuantizedFeaturesInfo->CatFeaturesPerfectHash;

        TArrayRef<ui32> dstBinsValue;
        if (dstBins.Defined()) {
            dstBinsValue = **dstBins;
            CheckDataSize(
                dstBinsValue.size(),
                (size_t)hashedCatArraySubset.Size(),
                /*dataName*/ "dstBins",
                /*dataCanBeEmpty*/ false,
                /*expectedSizeName*/ "hashedCatArraySubset",
                /*internalCheck*/ true
            );
        }

        TMap<ui32, ui32> perfectHashMap;
        {
            TWriteGuard guard(QuantizedFeaturesInfo->GetRWMutex());
            if (!featuresHash.HasHashInRam) {
                featuresHash.Load();
            }
            perfectHashMap.swap(featuresHash.FeaturesPerfectHash[*catFeatureIdx]);
        }

        constexpr size_t MAX_UNIQ_CAT_VALUES =
            static_cast<size_t>(Max<ui32>()) + ((sizeof(size_t) > sizeof(ui32)) ? 1 : 0);

        hashedCatArraySubset.ForEach(
            [&] (ui32 idx, ui32 hashedCatValue) {
                auto it = perfectHashMap.find(hashedCatValue);
                if (it == perfectHashMap.end()) {
                    CB_ENSURE(
                        perfectHashMap.size() != MAX_UNIQ_CAT_VALUES,
                        "Error: categorical feature with id #" << *catFeatureIdx
                        << " has more than " << MAX_UNIQ_CAT_VALUES
                        << " unique values, which is currently unsupported"
                    );
                    ui32 bin = (ui32)perfectHashMap.size();
                    if (dstBins) {
                        dstBinsValue[idx] = bin;
                    }
                    perfectHashMap.emplace_hint(it, hashedCatValue, bin);
                } else if (dstBins) {
                    dstBinsValue[idx] = it->second;
                }
            }
        );

        {
            TWriteGuard guard(QuantizedFeaturesInfo->GetRWMutex());
            auto& uniqValuesCounts = featuresHash.CatFeatureUniqValuesCountsVector[*catFeatureIdx];
            if (!uniqValuesCounts.OnAll) {
                uniqValuesCounts.OnLearnOnly = perfectHashMap.size();
            }
            uniqValuesCounts.OnAll = perfectHashMap.size();
            featuresHash.FeaturesPerfectHash[*catFeatureIdx].swap(perfectHashMap);
        }
    }

}
