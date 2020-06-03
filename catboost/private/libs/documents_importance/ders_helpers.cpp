#include "ders_helpers.h"

#include <catboost/private/libs/algo/approx_updater_helpers.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>
#include <catboost/libs/helpers/exception.h>

#include <library/cpp/fast_exp/fast_exp.h>

#include <util/generic/cast.h>
#include <util/system/yassert.h>

#include <functional>


template <typename TError>
void EvaluateDerivativesForError(
    const TVector<double>& approxes,
    TConstArrayRef<float> target,
    ELossFunction lossFunction,
    ELeavesEstimation leafEstimationMethod,
    TVector<double>* firstDerivatives,
    TVector<double>* secondDerivatives,
    TVector<double>* thirdDerivatives
) {
    const bool isStoreExpApprox = IsStoreExpApprox(lossFunction);
    ui32 docCount = SafeIntegerCast<ui32>(target.size());

    TVector<double> expApproxes;
    if (isStoreExpApprox) {
        expApproxes.resize(docCount);
        for (ui32 docId = 0; docId < docCount; ++docId) {
            expApproxes[docId] = fast_exp(approxes[docId]);
        }
    }
    const TVector<double>& approxesRef = isStoreExpApprox ? expApproxes : approxes;

    TError error(isStoreExpApprox);
    CheckDerivativeOrderForObjectImportance(error.GetMaxSupportedDerivativeOrder(), leafEstimationMethod);
    TVector<TDers> derivatives(docCount);

    Y_ASSERT(error.GetErrorType() == EErrorType::PerObjectError);
    error.CalcDersRange(
        /*start=*/0,
        docCount,
        /*calcThirdDer=*/thirdDerivatives != nullptr,
        approxesRef.data(),
        /*approxDeltas=*/nullptr,
        target.data(),
        /*weights=*/nullptr,
        derivatives.data()
    );

    for (ui32 docId = 0; docId < docCount; ++docId) {
        if (firstDerivatives) {
            (*firstDerivatives)[docId] = -derivatives[docId].Der1;
        }
        if (secondDerivatives) {
            (*secondDerivatives)[docId] = -derivatives[docId].Der2;
        }
        if (thirdDerivatives) {
            (*thirdDerivatives)[docId] = -derivatives[docId].Der3;
        }
    }
}

using TEvaluateDerivativesFunc = std::function<void(
    const TVector<double>& approxes,
    TConstArrayRef<float> target,
    ELossFunction lossFunction,
    ELeavesEstimation leafEstimationMethod,
    TVector<double>* firstDerivatives,
    TVector<double>* secondDerivatives,
    TVector<double>* thirdDerivatives
)>;

static TEvaluateDerivativesFunc GetEvaluateDerivativesFunc(ELossFunction lossFunction) {
    switch (lossFunction) {
        case ELossFunction::Logloss:
        case ELossFunction::CrossEntropy:
            return EvaluateDerivativesForError<TCrossEntropyError>;
        case ELossFunction::RMSE:
            return EvaluateDerivativesForError<TRMSEError>;
        case ELossFunction::MAE:
        case ELossFunction::Quantile:
            return EvaluateDerivativesForError<TQuantileError>;
        case ELossFunction::Expectile:
            return EvaluateDerivativesForError<TExpectileError>;
        case ELossFunction::LogLinQuantile:
            return EvaluateDerivativesForError<TLogLinQuantileError>;
        case ELossFunction::MAPE:
            return EvaluateDerivativesForError<TMAPError>;
        case ELossFunction::Poisson:
            return EvaluateDerivativesForError<TPoissonError>;
        default:
            CB_ENSURE(false, "Error function " + ToString(lossFunction) + " is not supported yet in ostr mode");
    }
}

void EvaluateDerivatives(
    ELossFunction lossFunction,
    ELeavesEstimation leafEstimationMethod,
    const TVector<double>& approxes,
    TConstArrayRef<float> target,
    TVector<double>* firstDerivatives,
    TVector<double>* secondDerivatives,
    TVector<double>* thirdDerivatives
) {
    auto evaluateDerivativesFunc = GetEvaluateDerivativesFunc(lossFunction);
    evaluateDerivativesFunc(
        approxes,
        target,
        lossFunction,
        leafEstimationMethod,
        firstDerivatives,
        secondDerivatives,
        thirdDerivatives
    );
}
