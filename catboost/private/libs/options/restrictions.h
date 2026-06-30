#pragma once

#include "enums.h"

#include <catboost/private/libs/ctr_description/ctr_type.h>

#include <util/system/compiler.h>
#include <util/system/types.h>

constexpr ui32 GetMaxBinCount() {
    return 65535;
}

constexpr ui32 GetMaxTreeDepth() {
    return 16;
}

constexpr bool IsSupportedCtrType(ETaskType taskType, ECtrType ctrType) {
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
    Y_UNREACHABLE();
}


//for performance reasons max tree-ctr binarization is compile-time constant on GPU
constexpr ui32 GetMaxTreeCtrBinarizationForGpu() {
    return 15;
}


//CPU restriction
using TIndexType = ui32;
constexpr int CB_THREAD_LIMIT = 128;
