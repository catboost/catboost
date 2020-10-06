#include "apply_result_iterator.h"

#include <catboost/private/libs/algo/apply.h>

#include <util/generic/cast.h>


TApplyResultIterator::TApplyResultIterator(
    const TFullModel& model,
    NCB::TRawObjectsDataProviderPtr rawObjectsDataProvider,
    EPredictionType predictionType,
    i32 threadCount
) throw(yexception)
    : ApplyResult(
          ApplyModelMulti(
              model,
              *rawObjectsDataProvider,
              /*verbose*/ false,
              predictionType,
              /*begin*/ 0,
              /*end*/ 0,
              SafeIntegerCast<int>(threadCount)
          )
      )
{}

