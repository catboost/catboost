#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/cuda/cuda_util/hash_helpers.h>
#include <catboost/libs/ctr_description/ctr_type.h>

#include <util/generic/vector.h>
#include <util/digest/multi.h>
#include <util/generic/algorithm.h>
#include <util/digest/city.h>
#include <util/generic/set.h>
#include <util/generic/map.h>
#include <library/binsaver/bin_saver.h>

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

constexpr inline bool IsPermutationDependentCtrType(const ECtrType& ctrType) {
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

inline yset<ECtrType> TakePermutationDependent(const yset<ECtrType>& types) {
    yset<ECtrType> result;
    for (auto type : types) {
        if (IsPermutationDependentCtrType(type)) {
            result.insert(type);
        }
    }
    return result;
}

inline yset<ECtrType> TakePermutationIndependent(const yset<ECtrType>& types) {
    yset<ECtrType> result;
    for (auto type : types) {
        if (!IsPermutationDependentCtrType(type)) {
            result.insert(type);
        }
    }
    return result;
}

inline yset<ECtrType> GetPermutationIndependentCtrs() {
    return {ECtrType::FeatureFreq};
}

struct TCtrConfig {
    ECtrType Type = ECtrType::FloatTargetMeanValue;
    yvector<float> Prior;
    ui32 ParamId = 0;

    ui64 GetHash() const {
        return MultiHash(Type, VecCityHash(Prior), ParamId);
    }

    bool operator<(const TCtrConfig& other) const {
        return std::tie(Type, Prior, ParamId) <
               std::tie(other.Type, other.Prior, other.ParamId);
    }

    bool operator==(const TCtrConfig& other) const {
        return std::tie(Type, Prior, ParamId) == std::tie(other.Type, other.Prior, other.ParamId);
    }

    bool operator!=(const TCtrConfig& other) const {
        return !(*this == other);
    }

    SAVELOAD(Type, Prior, ParamId);
};

inline TCtrConfig RemovePrior(const TCtrConfig& ctrConfig) {
    TCtrConfig result = ctrConfig;
    result.Prior.clear();
    return result;
}

inline ymap<TCtrConfig, yvector<TCtrConfig>> CreateEqualUpToPriorCtrsGroupping(const yvector<TCtrConfig>& configs) {
    ymap<TCtrConfig, yvector<TCtrConfig>> result;
    for (auto& config : configs) {
        TCtrConfig withoutPriorConfig = RemovePrior(config);
        result[withoutPriorConfig].push_back(config);
    }
    return result;
}

// equal configs factor
inline bool IsEqualUpToPrior(const TCtrConfig& left, const TCtrConfig& right) {
    return (left.ParamId == right.ParamId) && (left.Type == right.Type);
}

inline float GetPriorsSum(const TCtrConfig& config) {
    float total = 0;
    for (auto& val : config.Prior) {
        total += val;
    }
    return total;
}

inline float GetNumeratorShift(const TCtrConfig& config) {
    switch (config.Type) {
        case ECtrType::FloatTargetMeanValue:
        case ECtrType::FeatureFreq: {
            return config.Prior.at(0);
        }
        case ECtrType::Buckets:
        case ECtrType::Borders: {
            return config.Prior.at(config.ParamId);
        }
        default: {
            ythrow TCatboostException() << "unknown type";
        }
    }
}

inline float GetDenumeratorShift(const TCtrConfig& config) {
    switch (config.Type) {
        case ECtrType::FloatTargetMeanValue:
        case ECtrType::FeatureFreq: {
            return config.Prior.at(1);
        }
        case ECtrType::Buckets:
        case ECtrType::Borders: {
            return GetPriorsSum(config);
        }
        default: {
            ythrow TCatboostException() << "unknown type";
        }
    }
}
inline TCtrConfig CreateCtrConfigForFeatureFreq(float prior,
                                                ui32 uniqueValues) {
    TCtrConfig config;
    config.Type = ECtrType::FeatureFreq;
    config.ParamId = 0;
    config.Prior = {prior, (float)0.5 * uniqueValues};
    return config;
}

//dim without total stat
inline ui32 SufficientSpaceDim(const TCtrConfig& config) {
    Y_ENSURE(config.Prior.size(), "Error: provide prior");
    switch (config.Type) {
        case ECtrType::Buckets:
        case ECtrType::Borders: {
            return (ui32)(config.Prior.size() - 1);
        }
        case ECtrType::FloatTargetMeanValue: {
            Y_ENSURE(config.Prior.size() == 2);
            return 1;
        }
        case ECtrType::FeatureFreq: {
            return config.Prior.size() - 1;
        }
        default: {
            ythrow TCatboostException() << "unknown ctr type type " << config.Type;
        }
    }
}
