#pragma once

#include <util/system/types.h>

struct TCvDataPartitionParams {
    int FoldIdx = -1;
    size_t FoldCount = 0;
    bool Inverted = false;
    int RandSeed = 0;
};

struct TCrossValidationParams {
    ui64 FoldCount = 0;
    bool Inverted = false;
    int PartitionRandSeed = 0;
    bool Shuffle = true;
    bool Stratified = false;
};
