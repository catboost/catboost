#pragma once

#include <util/generic/yexception.h>


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
            ythrow yexception() << "Unknown ctr type " << ctr;
        }
    }
}
