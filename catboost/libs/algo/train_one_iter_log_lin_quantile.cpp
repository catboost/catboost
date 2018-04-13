#include "train_templ.h"

template <>
TLogLinQuantileError BuildError<TLogLinQuantileError>(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&) {
    auto lossParams = params.LossFunctionDescription->GetLossParams();
    if (lossParams.empty()) {
        return TLogLinQuantileError(IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction()));
    } else {
        CB_ENSURE(lossParams.begin()->first == "alpha", "Invalid loss description" << ToString(params.LossFunctionDescription.Get()));
        return TLogLinQuantileError(FromString<float>(lossParams["alpha"]), IsStoreExpApprox(params.LossFunctionDescription->GetLossFunction()));
    }
}

template void TrainOneIter<TLogLinQuantileError>(const TDataset&, const TDataset*, TLearnContext*);
