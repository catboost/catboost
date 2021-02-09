#include "apply_result_iterator.h"

#include <catboost/private/libs/algo/apply.h>


TApplyResultIterator::TApplyResultIterator(
    const TFullModel& model,
    NCB::TRawObjectsDataProviderPtr rawObjectsDataProvider,
    EPredictionType predictionType,
    NPar::TLocalExecutor* localExecutor
) throw(yexception)
    : ApplyResult(
          ApplyModelMulti(
              model,
              *rawObjectsDataProvider,
              predictionType,
              /*begin*/ 0,
              /*end*/ 0,
              localExecutor
          )
      )
{}

