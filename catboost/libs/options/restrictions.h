#pragma once

#include "enums.h"

#include <util/system/types.h>
#include <catboost/libs/ctr_description/ctr_type.h>

inline constexpr ui32 GetMaxBinCount() {
    return 255;
}

inline constexpr ui32 GetMaxTreeDepth() {
    return 16;
}

inline bool IsSupportedCtrType(ETaskType taskType, ECtrType ctrType) {
    switch (taskType) {
        case ETaskType::CPU: {
            switch (ctrType) {
                case ECtrType::Borders:
                case ECtrType::Buckets:
                case ECtrType::BinarizedTargetMeanValue:
                case ECtrType::Counter: {
                    return true;
                }
                default: {
                    return false;
                }
            }
        }
        case ETaskType::GPU: {
            switch (ctrType) {
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
    }
}


//for performance reasons max tree-ctr binarization is compile-time constant on GPU
inline constexpr ui32 GetMaxTreeCtrBinarizationForGpu() {
    return 15;
}


//CPU restriction
using TIndexType = ui32;
constexpr int CB_THREAD_LIMIT = 128;
