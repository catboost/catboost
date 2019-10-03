#pragma once

#include "ctr_type.h"

#include <catboost/libs/helpers/hash.h>

#include <util/digest/multi.h>
#include <util/generic/vector.h>
#include <util/system/types.h>
#include <util/str_stl.h>
#include <util/ysaveload.h>

#include <tuple>


namespace NCB {

    struct TCtrConfig {
        ECtrType Type = ECtrType::Borders;
        TVector<float> Prior;
        ui32 ParamId = 0;
        ui32 CtrBinarizationConfigId = 0;

        ui64 GetHash() const {
            return MultiHash(Type, VecCityHash(Prior), ParamId, CtrBinarizationConfigId);
        }

        bool operator<(const TCtrConfig& other) const {
            return std::tie(Type, Prior, ParamId, CtrBinarizationConfigId) <
                   std::tie(other.Type, other.Prior, other.ParamId, other.CtrBinarizationConfigId);
        }

        bool operator==(const TCtrConfig& other) const {
            return std::tie(Type, Prior, ParamId, CtrBinarizationConfigId) == std::tie(other.Type, other.Prior, other.ParamId, other.CtrBinarizationConfigId);
        }

        bool operator!=(const TCtrConfig& other) const {
            return !(*this == other);
        }

        Y_SAVELOAD_DEFINE(Type, Prior, ParamId, CtrBinarizationConfigId);
    };

}

template <>
struct THash<NCB::TCtrConfig> {
    inline size_t operator()(const NCB::TCtrConfig& config) const {
        return config.GetHash();
    }
};

