#include "cat_feature_perfect_hash_helper.h"

#include "util.h"

#include <util/system/guard.h>
#include <util/system/yassert.h>
#include <util/generic/cast.h>
#include <util/generic/map.h>

#include <util/generic/ylimits.h>

#include <utility>


namespace NCB {

    void TCatFeaturesPerfectHashHelper::UpdatePerfectHashAndMaybeQuantize(
        const TCatFeatureIdx catFeatureIdx,
        const ITypedArraySubset<ui32>& hashedCatArraySubset,
        bool mapMostFrequentValueTo0,
        TMaybe<TDefaultValue<ui32>> hashedCatDefaultValue,
        TMaybe<float> quantizedDefaultBinFraction,
        TMaybe<TArrayRef<ui32>*> dstBins
    ) {
        QuantizedFeaturesInfo->CheckCorrectPerTypeFeatureIdx(catFeatureIdx);
        auto& featuresHash = QuantizedFeaturesInfo->CatFeaturesPerfectHash;

        TArrayRef<ui32> dstBinsValue;
        if (dstBins.Defined()) {
            dstBinsValue = **dstBins;
            CheckDataSize(
                dstBinsValue.size(),
                (size_t)hashedCatArraySubset.GetSize(),
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
            perfectHashMap = std::move(featuresHash.FeaturesPerfectHash[*catFeatureIdx]);
        }

        // if perfectHashMap is already non-empty existing mapping can't be modified
        const bool perfectHashMapWasEmptyBeforeUpdate = perfectHashMap.Empty();

        constexpr size_t MAX_UNIQ_CAT_VALUES =
            static_cast<size_t>(Max<ui32>()) + ((sizeof(size_t) > sizeof(ui32)) ? 1 : 0);

        ui32 datasetSize = hashedCatArraySubset.GetSize();

        if (hashedCatDefaultValue) {
            datasetSize += SafeIntegerCast<ui32>(hashedCatDefaultValue->Count);

            if (perfectHashMapWasEmptyBeforeUpdate) {
                ui32 bin = 0;

                const float defaultValueFraction
                    = float(hashedCatDefaultValue->Count) / float(datasetSize);

                if (quantizedDefaultBinFraction &&
                    (defaultValueFraction >= (*quantizedDefaultBinFraction)))
                {
                    perfectHashMap.DefaultMap = TCatFeaturePerfectHashDefaultValue{
                        hashedCatDefaultValue->Value,
                        TValueWithCount{bin, (ui32)hashedCatDefaultValue->Count},
                        defaultValueFraction
                    };
                } else {
                    perfectHashMap.Map.emplace(
                        hashedCatDefaultValue->Value,
                        TValueWithCount{bin, (ui32)hashedCatDefaultValue->Count}
                    );
                }
            } else {
                // cannot update DefaultMap mapping, only insert as a regular element

                if (perfectHashMap.DefaultMap &&
                    (perfectHashMap.DefaultMap->SrcValue == hashedCatDefaultValue->Value))
                {
                    perfectHashMap.DefaultMap->DstValueWithCount.Count
                        += (ui32)hashedCatDefaultValue->Count;
                } else {
                    auto it = perfectHashMap.Map.find(hashedCatDefaultValue->Value);
                    if (it == perfectHashMap.Map.end()) {
                        CB_ENSURE(
                            perfectHashMap.Map.size() != MAX_UNIQ_CAT_VALUES,
                            "Error: categorical feature with id #" << *catFeatureIdx
                            << " has more than " << MAX_UNIQ_CAT_VALUES
                            << " unique values, which is currently unsupported"
                        );
                        const ui32 bin = (ui32)perfectHashMap.GetSize();
                        perfectHashMap.Map.emplace_hint(
                            it,
                            hashedCatDefaultValue->Value,
                            TValueWithCount{bin, (ui32)hashedCatDefaultValue->Count}
                        );
                    } else {
                        it->second.Count += (ui32)hashedCatDefaultValue->Count;
                    }
                }
            }
        }

        auto processNonDefaultValue = [&, dstBins, dstBinsValue] (ui32 idx, ui32 hashedCatValue) {
            auto it = perfectHashMap.Map.find(hashedCatValue);
            if (it == perfectHashMap.Map.end()) {
                CB_ENSURE(
                    perfectHashMap.Map.size() != MAX_UNIQ_CAT_VALUES,
                    "Error: categorical feature with id #" << *catFeatureIdx
                    << " has more than " << MAX_UNIQ_CAT_VALUES
                    << " unique values, which is currently unsupported"
                );
                const ui32 bin = (ui32)perfectHashMap.GetSize();
                if (dstBins) {
                    dstBinsValue[idx] = bin;
                }
                perfectHashMap.Map.emplace_hint(it, hashedCatValue, TValueWithCount{bin, 1});
            } else {
                if (dstBins) {
                    dstBinsValue[idx] = it->second.Value;
                }
                ++(it->second.Count);
            }
        };

        if (perfectHashMap.DefaultMap) {
            const TCatFeaturePerfectHashDefaultValue defaultMap = *perfectHashMap.DefaultMap;
            const ui32 defaultHashedCatValue = defaultMap.SrcValue;
            const ui32 defaultMappedValue = defaultMap.DstValueWithCount.Value;
            hashedCatArraySubset.ForEach(
                [&, defaultHashedCatValue, defaultMappedValue] (ui32 idx, ui32 hashedCatValue) {
                    if (hashedCatValue == defaultHashedCatValue) {
                        if (dstBins) {
                            dstBinsValue[idx] = defaultMappedValue;
                        }
                    } else {
                        processNonDefaultValue(idx, hashedCatValue);
                    }
                }
            );
        } else {
            hashedCatArraySubset.ForEach(processNonDefaultValue);
        }

        if (perfectHashMapWasEmptyBeforeUpdate &&
            !perfectHashMap.DefaultMap && // if default map exists it is already mapped to 0
            (quantizedDefaultBinFraction || mapMostFrequentValueTo0))
        {
            auto iter = perfectHashMap.Map.begin();
            auto iterEnd = perfectHashMap.Map.end();

            ui32 mostFrequentSrcValue = iter->first;
            TValueWithCount* mappedForMostFrequent = &(iter->second);
            TValueWithCount* mappedTo0 = (iter->second.Value == 0) ? &(iter->second) : nullptr;
            for (++iter; iter != iterEnd; ++iter) {
                TValueWithCount* mapped = &(iter->second);
                if (mapped->Count > mappedForMostFrequent->Count) {
                    mostFrequentSrcValue = iter->first;
                    mappedForMostFrequent = mapped;
                }
                if (mapped->Value == 0) {
                    mappedTo0 = mapped;
                }
            }
            if (mapMostFrequentValueTo0 && (mappedTo0 != mappedForMostFrequent)) {
                CB_ENSURE(mappedTo0, "No value mapped to 0");

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

            const float mostFrequentValueFraction
                = float(mappedForMostFrequent->Count) / float(datasetSize);

            if (quantizedDefaultBinFraction &&
                (mostFrequentValueFraction >= (*quantizedDefaultBinFraction)))
            {
                // move from Map to DefaultMap
                perfectHashMap.DefaultMap = TCatFeaturePerfectHashDefaultValue{
                    mostFrequentSrcValue,
                    TValueWithCount{mappedForMostFrequent->Value, (ui32)mappedForMostFrequent->Count},
                    mostFrequentValueFraction
                };
                perfectHashMap.Map.erase(mostFrequentSrcValue);
            }
        }

        {
            TWriteGuard guard(QuantizedFeaturesInfo->GetRWMutex());
            auto& uniqValuesCounts = featuresHash.CatFeatureUniqValuesCountsVector[*catFeatureIdx];
            if (!uniqValuesCounts.OnAll) {
                uniqValuesCounts.OnLearnOnly = perfectHashMap.GetSize();
            }
            uniqValuesCounts.OnAll = perfectHashMap.GetSize();
            featuresHash.FeaturesPerfectHash[*catFeatureIdx] = std::move(perfectHashMap);
        }
    }

}
