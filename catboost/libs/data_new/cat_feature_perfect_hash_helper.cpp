#include "cat_feature_perfect_hash_helper.h"

#include <util/system/guard.h>
#include <util/generic/map.h>

#include <util/generic/ylimits.h>


namespace NCB {

    void TCatFeaturesPerfectHashHelper::UpdatePerfectHashAndMaybeQuantize(
        const TCatFeatureIdx catFeatureIdx,
        TMaybeOwningArraySubset<ui32, ui32> hashedCatArraySubset,
        TMaybe<TVector<ui32>*> dstBins
    ) {
        QuantizedFeaturesInfo->CheckCorrectPerTypeFeatureIdx(catFeatureIdx);
        auto& featuresHash = QuantizedFeaturesInfo->CatFeaturesPerfectHash;

        TVector<ui32>* dstBinsPtr = nullptr;
        if (dstBins.Defined()) {
            dstBinsPtr = *dstBins;
            dstBinsPtr->yresize((int)hashedCatArraySubset.Size());
        }

        TMap<int, ui32> perfectHashMap;
        {
            TGuard<TAdaptiveLock> guard(UpdateLock);
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
                    if (dstBinsPtr) {
                        (*dstBinsPtr)[idx] = perfectHashMap.size();
                    }
                    perfectHashMap.emplace_hint(it, hashedCatValue, perfectHashMap.size());
                } else if (dstBinsPtr) {
                    (*dstBinsPtr)[idx] = it->second;
                }
            }
        );

        if (perfectHashMap.size() > 1) {
            TGuard<TAdaptiveLock> guard(UpdateLock);
            const ui32 uniqueValues = perfectHashMap.size();
            featuresHash.FeaturesPerfectHash[*catFeatureIdx].swap(perfectHashMap);
            featuresHash.CatFeatureUniqueValues[*catFeatureIdx] = uniqueValues;
        }
    }

}
