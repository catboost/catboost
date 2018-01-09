#include "train_templ.h"

namespace {
template <>
inline TQuantileError BuildError<TQuantileError>(const NCatboostOptions::TCatBoostOptions& params, const TMaybe<TCustomObjectiveDescriptor>&) {
    auto lossParams = params.LossFunctionDescription->GetLossParams();
    if (lossParams.empty()) {
        return TQuantileError(IsStoreExpApprox(params));
    } else {
        CB_ENSURE(lossParams.begin()->first == "alpha", "Invalid loss description" << ToString(params.LossFunctionDescription.Get()));
        return TQuantileError(FromString<float>(lossParams["alpha"]), IsStoreExpApprox(params));
    }
}
}

template void TrainOneIter<TQuantileError>(const TTrainData&, TLearnContext*);
