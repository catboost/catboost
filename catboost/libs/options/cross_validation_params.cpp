#include "cross_validation_params.h"

#include <catboost/libs/helpers/exception.h>


void TCvDataPartitionParams::Check() const {
    CB_ENSURE(FoldCount, "FoldCount is 0");
    CB_ENSURE(FoldIdx < FoldCount, "FoldIdx (" << FoldIdx << ") >= FoldCount (" << FoldCount << ')');
}
