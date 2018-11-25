#include "train_templ.h"

template <>
TLqError BuildError<TLqError>(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&) {
    double q = NCatboostOptions::GetLqParam(params.LossFunctionDescription);
    return {q, IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction())};
}

template void TrainOneIter<TLqError>(const TDataset&, const TDatasetPtrs&, TLearnContext*);
