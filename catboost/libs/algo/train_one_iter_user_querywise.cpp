#include "train_templ.h"

namespace {
template <>
inline TUserDefinedQuerywiseError BuildError<TUserDefinedQuerywiseError>(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&) {
    return TUserDefinedQuerywiseError(params.LossFunctionDescription->GetLossParams(), IsStoreExpApprox(params));
}
}

template void TrainOneIter<TUserDefinedQuerywiseError>(const TTrainData&, const TTrainData*, TLearnContext*);
