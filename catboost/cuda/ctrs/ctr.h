#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/ctr_description/ctr_config.h>
#include <catboost/private/libs/ctr_description/ctr_type.h>

#include <util/generic/vector.h>
#include <util/generic/algorithm.h>
#include <util/digest/city.h>
#include <util/generic/set.h>
#include <util/generic/map.h>

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

    inline NCB::TCtrConfig RemovePrior(const NCB::TCtrConfig& ctrConfig) {
        NCB::TCtrConfig result = ctrConfig;
        result.Prior.clear();
        return result;
    }

    inline TMap<NCB::TCtrConfig, TVector<NCB::TCtrConfig>> CreateEqualUpToPriorAndBinarizationCtrsGroupping(const TVector<NCB::TCtrConfig>& configs) {
        TMap<NCB::TCtrConfig, TVector<NCB::TCtrConfig>> result;
        for (auto& config : configs) {
            NCB::TCtrConfig withoutPriorConfig = RemovePrior(config);
            withoutPriorConfig.CtrBinarizationConfigId = -1;
            result[withoutPriorConfig].push_back(config);
        }
        return result;
    }

    // equal configs factor
    inline bool IsEqualUpToPriorAndBinarization(const NCB::TCtrConfig& left, const NCB::TCtrConfig& right) {
        return (left.ParamId == right.ParamId) && (left.Type == right.Type);
    }

    inline float GetNumeratorShift(const NCB::TCtrConfig& config) {
        return config.Prior.at(0);
    }

    inline float GetDenumeratorShift(const NCB::TCtrConfig& config) {
        return config.Prior.at(1);
    }

    inline NCB::TCtrConfig CreateCtrConfigForFeatureFreq(float prior,
                                                         ui32 uniqueValues) {
        NCB::TCtrConfig config;
        config.Type = ECtrType::FeatureFreq;
        config.ParamId = 0;
        config.Prior = {prior, (float)0.5 * uniqueValues};
        return config;
    }

}
