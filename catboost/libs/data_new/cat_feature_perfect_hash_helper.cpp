#include "cat_feature_perfect_hash_helper.h"

#include "util.h"

#include <util/system/guard.h>
#include <util/system/yassert.h>
#include <util/generic/map.h>

#include <util/generic/ylimits.h>

#include <utility>


namespace NCB {

    void TCatFeaturesPerfectHashHelper::UpdatePerfectHashAndMaybeQuantize(
        const TCatFeatureIdx catFeatureIdx,
        TMaybeOwningConstArraySubset<ui32, ui32> hashedCatArraySubset,
        bool mapMostFrequentValueTo0,
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

        TCatFeaturePerfectHash perfectHashMap;
        {
            TWriteGuard guard(QuantizedFeaturesInfo->GetRWMutex());
            if (!featuresHash.HasHashInRam) {
                featuresHash.Load();
            }
            perfectHashMap.swap(featuresHash.FeaturesPerfectHash[*catFeatureIdx]);
        }

        // if perfectHashMap is already non-empty existing mapping can't be modified
        mapMostFrequentValueTo0 = mapMostFrequentValueTo0 && perfectHashMap.empty();

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
                    perfectHashMap.emplace_hint(it, hashedCatValue, TValueWithCount{bin, 1});
                } else {
                    if (dstBins) {
                        dstBinsValue[idx] = it->second.Value;
                    }
                    ++(it->second.Count);
                }
            }
        );

        if (mapMostFrequentValueTo0 && !perfectHashMap.empty()) {
            auto iter = perfectHashMap.begin();
            auto iterEnd = perfectHashMap.end();

            TValueWithCount* mappedForMostFrequent = &(iter->second);
            TValueWithCount* mappedTo0 = (iter->second.Value == 0) ? &(iter->second) : nullptr;
            for (++iter; iter != iterEnd; ++iter) {
                TValueWithCount* mapped = &(iter->second);
                if (mapped->Count > mappedForMostFrequent->Count) {
                    mappedForMostFrequent = mapped;
                }
                if (mapped->Value == 0) {
                    mappedTo0 = mapped;
                }
            }
            if (mappedTo0 != mappedForMostFrequent) {
                Y_VERIFY(mappedTo0, "No value mapped to 0");

                if (dstBins) {
                    // values for 0 and MostFrequent need to be swapped
                    const ui32 valueToSwapWith0 = mappedForMostFrequent->Value;

                    for (auto& dstBin : dstBinsValue) {
                        if (dstBin == 0) {
                            dstBin = valueToSwapWith0;
                        } else if (dstBin == valueToSwapWith0) {
                            dstBin = 0;
                        }
                    }
                }

                std::swap(mappedTo0->Value, mappedForMostFrequent->Value);
            }
        }

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
