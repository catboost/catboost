#include "train_templ.h"

template<>
TQuantileError BuildError<TQuantileError>(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&) {
    auto lossParams = params.LossFunctionDescription->GetLossParams();
    if (lossParams.empty()) {
        return TQuantileError(IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction()));
    } else {
        CB_ENSURE(lossParams.begin()->first == "alpha", "Invalid loss description" << ToString(params.LossFunctionDescription.Get()));
        return TQuantileError(FromString<float>(lossParams["alpha"]), IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction()));
    }
}

template void TrainOneIter<TQuantileError>(const TDataset&, const TDataset*, TLearnContext*);
