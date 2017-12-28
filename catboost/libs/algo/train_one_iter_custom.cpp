#include "train_templ.h"

namespace {
template <>
inline TCustomError BuildError<TCustomError>(const NCatboostOptions::TCatBoostOptions& params,
    const TMaybe<TCustomObjectiveDescriptor>& descriptor) {
    Y_ASSERT(descriptor.Defined());
    return TCustomError(params, descriptor);
}
}

template void TrainOneIter<TCustomError>(const TTrainData&, TLearnContext*);
