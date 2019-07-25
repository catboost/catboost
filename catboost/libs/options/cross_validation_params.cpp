#include "cross_validation_params.h"

#include <catboost/libs/helpers/exception.h>


void TCrossValidationParams::Check() const {
    if (customTrainSubsets.Defined() || customTestSubsets.Defined()) {
        CB_ENSURE(
            (customTrainSubsets.Defined() && customTestSubsets.Defined()),
            "Custom train and test folds must be either both defined or both undefined"
        );
        CB_ENSURE(
            (customTrainSubsets->size() == customTestSubsets->size()),
            "Custom train and test folds should be the same size"
        );
        CB_ENSURE(
            customTrainSubsets->size() == FoldCount,
            "FoldCount must be the same as size of customTrainSubsets"
        );
    }
    CB_ENSURE(FoldCount, "FoldCount is 0");
    CB_ENSURE(
        (MaxTimeSpentOnFixedCostRatio > 0.0) && (MaxTimeSpentOnFixedCostRatio < 1.0),
        "MaxTimeSpentOnFixedCostRatio should be within (0, 1) range, got " << MaxTimeSpentOnFixedCostRatio
        << " instead"
    );
}


void TCvDataPartitionParams::Check() const {
    TCrossValidationParams::Check();
    CB_ENSURE(FoldIdx < FoldCount, "FoldIdx (" << FoldIdx << ") >= FoldCount (" << FoldCount << ')');
}
