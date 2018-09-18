#pragma once

#include <catboost/libs/options/enums.h>
#include <catboost/libs/data_util/path_with_scheme.h>
#include <catboost/libs/data_util/line_data_reader.h>
#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/labels/external_label_helper.h>
#include <catboost/libs/model/model.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/string/builder.h>
#include <util/generic/vector.h>
#include <util/generic/map.h>
#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/stream/output.h>
#include <util/stream/file.h>
#include <util/string/iterator.h>
#include <util/string/cast.h>


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

TVector<TVector<double>> MakeExternalApprox(
    const TVector<TVector<double>>& internalApprox,
    const TExternalLabelsHelper& externalLabelsHelper);

TVector<TString> ConvertTargetToExternalName(
    const TVector<float>& target,
    const TExternalLabelsHelper& externalLabelsHelper);

TVector<TString> ConvertTargetToExternalName(
    const TVector<float>& target,
    const TFullModel& model);

TVector<TVector<double>> PrepareEval(
    const EPredictionType predictionType,
    const TVector<TVector<double>>& approx,
    NPar::TLocalExecutor* localExecutor);

TVector<TVector<double>> PrepareEval(
    const EPredictionType predictionType,
    const TVector<TVector<double>>& approx,
    int threadCount);
