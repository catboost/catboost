#pragma once

enum class ECtrType {
    Borders,
    Buckets,
    MeanValue,
    Counter,
    FeatureFreq // TODO(kirillovs, vitekmel): only for cuda models, remove after implementing proper ctr binarization
};
