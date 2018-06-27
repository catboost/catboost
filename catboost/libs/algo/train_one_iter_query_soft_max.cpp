#include "train_templ.h"

template<>
TQuerySoftMaxError BuildError<TQuerySoftMaxError>(
    const NCatboostOptions::TCatBoostOptions& params,
    const TMaybe<TCustomObjectiveDescriptor>&
) {
    const auto& lossFunctionDescription = params.LossFunctionDescription;
    const auto& lossParams = lossFunctionDescription->GetLossParams();
    CB_ENSURE(
        lossParams.empty() || lossParams.begin()->first == "lambda",
        "Invalid loss description" << ToString(lossFunctionDescription.Get())
    );

    const double lambdaReg = NCatboostOptions::GetQuerySoftMaxLambdaReg(lossFunctionDescription);
    const bool isStoreExpApprox = IsStoreExpApprox(lossFunctionDescription->GetLossFunction());
    return TQuerySoftMaxError(lambdaReg, isStoreExpApprox);
}

template void TrainOneIter<TQuerySoftMaxError>(const TDataset&, const TDatasetPtrs&, TLearnContext*);
