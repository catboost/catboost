#pragma once

#include "split_params.h"

#include <util/system/types.h>


struct TCrossValidationParams : public TSplitParams {
    ui32 FoldCount = 0;
    bool Inverted = false;
    double MaxTimeSpentOnFixedCostRatio = 0.05;
    ui32 DevMaxIterationsBatchSize = 100000; // useful primarily for tests

public:
    bool Initialized() const {
        return FoldCount != 0;
    }

    void Check() const;
};

struct TCvDataPartitionParams : public TCrossValidationParams {
    ui32 FoldIdx = 0;

public:
    void Check() const;
};
