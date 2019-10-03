#pragma once


struct TSplitParams {
    int PartitionRandSeed = 0;
    bool Shuffle = true;
    bool Stratified = false;
};

struct TTrainTestSplitParams : public TSplitParams {
    double TrainPart = 0.8;
};
