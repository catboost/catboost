#include "train_templ.h"

template <>
TUserDefinedQuerywiseError BuildError<TUserDefinedQuerywiseError>(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&) {
    return TUserDefinedQuerywiseError(params.LossFunctionDescription->GetLossParams(), IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction()));
}

template void TrainOneIter<TUserDefinedQuerywiseError>(const TDataset&, const TDataset*, TLearnContext*);
