#include "train_templ.h"

template <>
TUserDefinedPerObjectError BuildError<TUserDefinedPerObjectError>(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&) {
    return TUserDefinedPerObjectError(params.LossFunctionDescription->GetLossParams(), IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction()));
}

template void TrainOneIter<TUserDefinedPerObjectError>(const TDataset&, const TDataset*, TLearnContext*);
