#pragma once

#include "split_params.h"

#include <library/cpp/binsaver/bin_saver.h>

#include <util/system/types.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>

#include <catboost/private/libs/options/enums.h>

struct TCrossValidationParams : public TSplitParams {
    TCrossValidationParams() = default;
    ui32 FoldCount = 0;

    // customTrainSubsets and customTestSubsets must be either both defined or both undefined
    // and when defined, they should have same sizes
    TMaybe<TVector<TVector<ui32>>> customTrainSubsets = Nothing();
    TMaybe<TVector<TVector<ui32>>> customTestSubsets = Nothing();
    double MaxTimeSpentOnFixedCostRatio = 0.05;
    double MetricUpdateInterval = 0.5; // in seconds
    ui32 DevMaxIterationsBatchSize = 100000; // useful primarily for tests
    ECrossValidation Type = ECrossValidation::Classical;
    bool IsCalledFromSearchHyperparameters = false;
    bool ReturnModels = false;

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
    SAVELOAD(customTrainSubsets, customTestSubsets);
};
