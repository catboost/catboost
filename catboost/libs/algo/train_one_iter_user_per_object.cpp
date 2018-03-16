#include "train_templ.h"

namespace {
template <>
inline TUserDefinedPerObjectError BuildError<TUserDefinedPerObjectError>(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&) {
    return TUserDefinedPerObjectError(params.LossFunctionDescription->GetLossParams(), IsStoreExpApprox(params));
}
}

template void TrainOneIter<TUserDefinedPerObjectError>(const TTrainData&, const TTrainData*, TLearnContext*);
