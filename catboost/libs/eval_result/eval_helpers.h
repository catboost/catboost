#pragma once

#include <catboost/private/libs/labels/external_label_helper.h>
// TODO(kirillovs): remove this include and fix external code
#include <catboost/libs/model/eval_processing.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/fwd.h>
#include <util/generic/vector.h>

namespace NCB {

TVector<TVector<double>> PrepareEvalForInternalApprox(
    const EPredictionType prediction_type,
    const TFullModel& model,
    const TVector<TVector<double>>& approx,
    int threadCount);

TVector<TVector<double>> PrepareEvalForInternalApprox(
    const EPredictionType prediction_type,
    const TFullModel& model,
    const TVector<TVector<double>>& approx,
    NPar::ILocalExecutor* executor = nullptr);

bool IsMulticlass(const TVector<TVector<double>>& approx);

void MakeExternalApprox(
    const TVector<TVector<double>>& internalApprox,
    const TExternalLabelsHelper& externalLabelsHelper,
    TVector<TVector<double>>* resultApprox);

TVector<TVector<double>> MakeExternalApprox(
    const TVector<TVector<double>>& internalApprox,
    const TExternalLabelsHelper& externalLabelsHelper);

void PrepareEval(
    const EPredictionType predictionType,
    size_t ensemblesCount,
    const TString& lossFunctionName,
    const TVector<TVector<double>>& approx,
    NPar::ILocalExecutor* executor,
    TVector<TVector<double>>* result,
    double binClassLogitThreshold = DEFAULT_BINCLASS_LOGIT_THRESHOLD);

TVector<TVector<double>> PrepareEval(
    const EPredictionType predictionType,
    size_t ensemblesCount,
    const TString& lossFunctionName,
    const TVector<TVector<double>>& approx,
    NPar::ILocalExecutor* executor = nullptr,
    double binClassLogitThreshold = DEFAULT_BINCLASS_LOGIT_THRESHOLD);

TVector<TVector<double>> PrepareEval(
    const EPredictionType predictionType,
    size_t ensemblesCount,
    const TString& lossFunctionName,
    const TVector<TVector<double>>& approx,
    int threadCount,
    double binClassLogitThreshold = DEFAULT_BINCLASS_LOGIT_THRESHOLD);
using TColumnPrinterOuputType = std::variant<i64, ui64, double, float, TString>;

}

TVector<TVector<double>> CalcSoftmax(
    const TVector<TVector<double>>& approx,
    NPar::ILocalExecutor* executor);
