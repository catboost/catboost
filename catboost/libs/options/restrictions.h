#pragma once

#include <util/system/types.h>
#include <catboost/libs/ctr_description/ctr_type.h>

inline constexpr ui32 GetMaxBinCount() {
    return 255;
}

inline constexpr ui32 GetMaxTreeDepth() {
    return 16;
}

inline bool IsSupportedOnGpu(ECtrType ctrType) {
    switch (ctrType) {
        case ECtrType::Borders:
        case ECtrType::Buckets:
        case ECtrType::FeatureFreq:
        case ECtrType::FloatTargetMeanValue: {
            return true;
        }
        default: {
            return false;
        }
    }
}

//for performance reasons max tree-ctr binarization is compile-time constant on GPU
inline constexpr ui32 GetMaxTreeCtrBinarizationForGpu() {
    return 15;
}

inline bool IsSupportedOnCpu(ECtrType ctrType) {
    switch (ctrType) {
        case ECtrType::Borders:
        case ECtrType::Buckets:
        case ECtrType::Counter:
        case ECtrType::BinarizedTargetMeanValue: {
            return true;
        }
        default: {
            return false;
        }
    }
}

//CPU restriction
using TIndexType = ui32;
constexpr int CB_THREAD_LIMIT = 56;
