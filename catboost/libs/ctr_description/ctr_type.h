#pragma once

#include <catboost/libs/helpers/exception.h>


enum class ECtrType {
    Borders,
    Buckets,
    BinarizedTargetMeanValue,
    FloatTargetMeanValue,
    Counter,
    FeatureFreq, // TODO(kirillovs): only for cuda models, remove after implementing proper ctr binarization
    CtrTypesCount
};

inline bool NeedTargetClassifier(ECtrType ctr) {
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

inline bool IsPermutationDependentCtrType(ECtrType ctr) {
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

