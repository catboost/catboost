#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/cuda/utils/hash_helpers.h>
#include <catboost/libs/ctr_description/ctr_type.h>

#include <util/generic/vector.h>
#include <util/digest/multi.h>
#include <util/generic/algorithm.h>
#include <util/digest/city.h>
#include <util/generic/set.h>
#include <util/generic/map.h>
#include <util/ysaveload.h>

namespace NCatboostCuda {
    inline bool IsBinarizedTargetCtr(ECtrType type) {
        return type == ECtrType::Buckets || type == ECtrType::Borders;
    }

    inline bool IsFloatTargetCtr(ECtrType type) {
        return type == ECtrType::FloatTargetMeanValue;
    }

    inline bool IsCatFeatureStatisticCtr(ECtrType type) {
        return type == ECtrType::FeatureFreq;
    }

    inline bool IsBordersBasedCtr(ECtrType type) {
        return type == ECtrType::Borders;
    }

    inline bool IsSupportedCtrType(ECtrType type) {
        switch (type) {
            case ECtrType::Borders:
            case ECtrType::Buckets:
            case ECtrType::FloatTargetMeanValue:
            case ECtrType::FeatureFreq: {
                return true;
            }
            default: {
                return false;
            }
        }
    }

    inline bool IsPermutationDependentCtrType(const ECtrType& ctrType) {
        switch (ctrType) {
            case ECtrType::Buckets:
            case ECtrType::Borders:
            case ECtrType::FloatTargetMeanValue: {
                return true;
            }
            case ECtrType::FeatureFreq: {
                return false;
            }
            default: {
                ythrow TCatboostException() << "unknown ctr type type " << ctrType;
            }
        }
    }

    inline TSet<ECtrType> TakePermutationDependent(const TSet<ECtrType>& types) {
        TSet<ECtrType> result;
        for (auto type : types) {
            if (IsPermutationDependentCtrType(type)) {
                result.insert(type);
            }
        }
        return result;
    }

    inline TSet<ECtrType> TakePermutationIndependent(const TSet<ECtrType>& types) {
        TSet<ECtrType> result;
        for (auto type : types) {
            if (!IsPermutationDependentCtrType(type)) {
                result.insert(type);
            }
        }
        return result;
    }

    inline TSet<ECtrType> GetPermutationIndependentCtrs() {
        return {ECtrType::FeatureFreq};
    }

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

    inline TCtrConfig RemovePrior(const TCtrConfig& ctrConfig) {
        TCtrConfig result = ctrConfig;
        result.Prior.clear();
        return result;
    }

    inline TMap<TCtrConfig, TVector<TCtrConfig>> CreateEqualUpToPriorAndBinarizationCtrsGroupping(const TVector<TCtrConfig>& configs) {
        TMap<TCtrConfig, TVector<TCtrConfig>> result;
        for (auto& config : configs) {
            TCtrConfig withoutPriorConfig = RemovePrior(config);
            withoutPriorConfig.CtrBinarizationConfigId = -1;
            result[withoutPriorConfig].push_back(config);
        }
        return result;
    }

    // equal configs factor
    inline bool IsEqualUpToPriorAndBinarization(const TCtrConfig& left, const TCtrConfig& right) {
        return (left.ParamId == right.ParamId) && (left.Type == right.Type);
    }

    inline float GetNumeratorShift(const TCtrConfig& config) {
        return config.Prior.at(0);
    }

    inline float GetDenumeratorShift(const TCtrConfig& config) {
        return config.Prior.at(1);
    }

    inline TCtrConfig CreateCtrConfigForFeatureFreq(float prior,
                                                    ui32 uniqueValues) {
        TCtrConfig config;
        config.Type = ECtrType::FeatureFreq;
        config.ParamId = 0;
        config.Prior = {prior, (float)0.5 * uniqueValues};
        return config;
    }

}
