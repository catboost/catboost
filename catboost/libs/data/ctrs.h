#pragma once

#include "data_provider.h"

#include <catboost/private/libs/ctr_description/ctr_type.h>

#include <util/generic/fwd.h>
#include <util/generic/hash.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/digest/multi.h>
#include <util/system/types.h>

#include <tuple>


namespace NCB {

    struct TOnlineCtrIdx {
    public:
        i32 CatFeatureIdx = 0;
        i16 CtrIdx = 0;
        i16 TargetBorderIdx = 0;
        i16 PriorIdx = 0;

    public:
        bool operator==(const TOnlineCtrIdx& rhs) const {
            return std::tie(CatFeatureIdx, CtrIdx, TargetBorderIdx, PriorIdx) ==
                std::tie(rhs.CatFeatureIdx, rhs.CtrIdx, rhs.TargetBorderIdx, rhs.PriorIdx);
        }

        size_t GetHash() const noexcept {
            return MultiHash(CatFeatureIdx, CtrIdx, TargetBorderIdx, PriorIdx);
        }
    };

}

template <>
struct THash<NCB::TOnlineCtrIdx> {
    size_t operator()(const NCB::TOnlineCtrIdx& onlineCtrIdx) const noexcept {
        return onlineCtrIdx.GetHash();
    }
};

namespace NCB {

    struct TOnlineCtrUniqValuesCounts {
        i32 Count = 0;

        // Counter ctrs could have more values than other types when counter_calc_method == Full
        i32 CounterCount = 0;

    public:
        bool operator==(const TOnlineCtrUniqValuesCounts& rhs) const {
            return std::tie(Count, CounterCount) == std::tie(rhs.Count, rhs.CounterCount);
        }

        i32 GetMaxUniqueValueCount() const {
            return Max(Count, CounterCount);
        }
        i32 GetUniqueValueCountForType(ECtrType type) const {
            if (ECtrType::Counter == type) {
                return CounterCount;
            } else {
                return Count;
            }
        }
    };


    struct TPrecomputedOnlineCtrMetaData {
        THashMap<TOnlineCtrIdx, ui32> OnlineCtrIdxToFeatureIdx;
        THashMap<ui32, TOnlineCtrUniqValuesCounts> ValuesCounts; // [catFeatureIdx]

    public:
        bool operator==(const TPrecomputedOnlineCtrMetaData& rhs) const {
            return std::tie(OnlineCtrIdxToFeatureIdx, ValuesCounts)
                == std::tie(rhs.OnlineCtrIdxToFeatureIdx, rhs.ValuesCounts);
        }

        void Append(TPrecomputedOnlineCtrMetaData& add);

        // Use JSON as string to be able to use in JVM binding as well
        TString SerializeToJson() const;
        static TPrecomputedOnlineCtrMetaData DeserializeFromJson(
            const TString& serializedJson
        );
    };


    struct TPrecomputedOnlineCtrData {
        TPrecomputedOnlineCtrMetaData Meta;
        TEstimatedForCPUObjectsDataProviders DataProviders;
    };
}
