#include "ctr_type.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/yexception.h>

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
            ythrow TCatboostException() << "Unknown ctr type " << ctr;
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
            ythrow TCatboostException() << "Unknown ctr type " << ctr;
        }
    }
}
