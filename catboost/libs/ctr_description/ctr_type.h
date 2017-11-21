#pragma once

enum class ECtrType {
    Borders,
    Buckets,
    BinarizedTargetMeanValue,
    FloatTargetMeanValue,
    Counter,
    FeatureFreq, // TODO(kirillovs): only for cuda models, remove after implementing proper ctr binarization
    CtrTypesCount
};
