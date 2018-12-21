#pragma once

#include <util/system/types.h>


struct TCrossValidationParams {
    ui32 FoldCount = 0;
    bool Inverted = false;
    int PartitionRandSeed = 0;
    bool Shuffle = true;
    bool Stratified = false;
    ui32 IterationsBatchSize = 100;

public:
    bool Initialized() const {
        return FoldCount != 0;
    }
};

struct TCvDataPartitionParams : public TCrossValidationParams {
    ui32 FoldIdx = 0;

public:
    void Check() const;
};
