#include "cross_validation_params.h"

#include <catboost/libs/helpers/exception.h>


void TCrossValidationParams::Check() const {
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
