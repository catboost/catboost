#pragma once

#include <catboost/libs/labels/external_label_helper.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/options/enums.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>


void CalcSoftmax(const TVector<double>& approx, TVector<double>* softmax);

TVector<double> CalcSigmoid(const TVector<double>& approx);

TVector<TVector<double>> PrepareEvalForInternalApprox(
    const EPredictionType prediction_type,
    const TFullModel& model,
    const TVector<TVector<double>>& approx,
    int threadCount);

TVector<TVector<double>> PrepareEvalForInternalApprox(
    const EPredictionType prediction_type,
    const TFullModel& model,
    const TVector<TVector<double>>& approx,
    NPar::TLocalExecutor* localExecutor);


bool IsMulticlass(const TVector<TVector<double>>& approx);

void MakeExternalApprox(
    const TVector<TVector<double>>& internalApprox,
    const TExternalLabelsHelper& externalLabelsHelper,
    TVector<TVector<double>>* resultApprox);

TVector<TVector<double>> MakeExternalApprox(
    const TVector<TVector<double>>& internalApprox,
    const TExternalLabelsHelper& externalLabelsHelper);

TVector<TString> ConvertTargetToExternalName(
    const TVector<float>& target,
    const TExternalLabelsHelper& externalLabelsHelper);

TVector<TString> ConvertTargetToExternalName(
    const TVector<float>& target,
    const TFullModel& model);

void PrepareEval(
    const EPredictionType predictionType,
    const TVector<TVector<double>>& approx,
    NPar::TLocalExecutor* localExecutor,
    TVector<TVector<double>>* result);

TVector<TVector<double>> PrepareEval(
    const EPredictionType predictionType,
    const TVector<TVector<double>>& approx,
    NPar::TLocalExecutor* localExecutor);

TVector<TVector<double>> PrepareEval(
    const EPredictionType predictionType,
    const TVector<TVector<double>>& approx,
    int threadCount);
