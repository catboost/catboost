#pragma once


namespace NMonoForest {
    enum class EFeatureType {
        Float,
        OneHot
    };

    enum class EBinSplitType {
        TakeBin,
        TakeGreater
    };

    enum class ESplitValue {
        Zero,
        One
    };
}
