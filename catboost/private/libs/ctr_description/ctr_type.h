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

bool NeedTarget(ECtrType ctr);

bool NeedTargetClassifier(ECtrType ctr);

bool IsPermutationDependentCtrType(ECtrType ctr);


enum class ECtrHistoryUnit {
    Group,
    Sample
};
