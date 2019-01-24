#include "ctr_type.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/yexception.h>


bool NeedTarget(ECtrType ctr) {
    switch (ctr) {
        case ECtrType::Buckets:
        case ECtrType::Borders:
        case ECtrType::BinarizedTargetMeanValue:
        case ECtrType::FloatTargetMeanValue: {
            return true;
        }
        case ECtrType::FeatureFreq:
        case ECtrType::Counter: {
            return false;
        }
        default: {
            ythrow TCatBoostException() << "Unknown ctr type " << ctr;
        }
    }
}

bool NeedTargetClassifier(const ECtrType ctr) {
    switch (ctr) {
        case ECtrType::FeatureFreq:
        case ECtrType::Counter:
        case ECtrType::FloatTargetMeanValue: {
            return false;
        }
        case ECtrType::Buckets:
        case ECtrType::Borders:
        case ECtrType::BinarizedTargetMeanValue: {
            return true;
        }
        default: {
            ythrow TCatBoostException() << "Unknown ctr type " << ctr;
        }
    }
}

bool IsPermutationDependentCtrType(const ECtrType ctr) {
    switch (ctr) {
        case ECtrType::Buckets:
        case ECtrType::Borders:
        case ECtrType::FloatTargetMeanValue:
        case ECtrType::BinarizedTargetMeanValue: {
            return true;
        }
        case ECtrType::Counter:
        case ECtrType::FeatureFreq: {
            return false;
        }
        default: {
            ythrow TCatBoostException() << "Unknown ctr type " << ctr;
        }
    }
}
