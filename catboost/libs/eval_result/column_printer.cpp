#include "column_printer.h"

#include "eval_helpers.h"

#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/loss_description.h>
#include <util/generic/utility.h>
#include <util/string/builder.h>


namespace NCB {

    void PushBackEvalPrinters(
        const TVector<TVector<TVector<double>>>& rawValues,
        const EPredictionType predictionType,
        const TString& lossFunctionName,
        bool isMultiTarget,
        size_t ensemblesCount,
        const TExternalLabelsHelper& visibleLabelsHelper,
        TMaybe<std::pair<size_t, size_t>> evalParameters,
        TVector<THolder<IColumnPrinter>>* result,
        NPar::ILocalExecutor* executor,
        double binClassLogitThreshold
    ) {
        int begin = 0;
        const bool callMakeExternalApprox
            = visibleLabelsHelper.IsInitialized()
              && (visibleLabelsHelper.GetExternalApproxDimension() > 1)
              && !IsUncertaintyPredictionType(predictionType);
        for (const auto& raws : rawValues) {
            const auto& approx = callMakeExternalApprox ? MakeExternalApprox(raws, visibleLabelsHelper) : raws;
            auto approxes = PrepareEval(predictionType, ensemblesCount, lossFunctionName, approx, executor, binClassLogitThreshold);
            const auto& headers = CreatePredictionTypeHeader(
                approx.size(),
                isMultiTarget,
                predictionType,
                visibleLabelsHelper,
                lossFunctionName,
                ensemblesCount,
                begin,
                evalParameters.Get()
            );
            for (int i = 0; i < approxes.ysize(); ++i) {
                result->push_back(MakeHolder<TEvalPrinter>(
                    predictionType,
                    headers[i],
                    approxes[i],
                    visibleLabelsHelper
                ));
            }
            if (evalParameters) {
                begin += evalParameters->first;
            }
        }
    }

    TVector<TString> CreatePredictionTypeHeader(
        ui32 approxDimension,
        bool isMultiTarget,
        EPredictionType predictionType,
        const TExternalLabelsHelper& visibleLabelsHelper,
        const TString& lossFunctionName,
        size_t ensemblesCount,
        ui32 startTreeIndex,
        std::pair<size_t, size_t>* evalParameters) {

        const ui32 classCount = (predictionType == EPredictionType::Class) ? 1 : approxDimension;
        TVector<TString> headers;
        bool isUncertainty = IsUncertaintyPredictionType(predictionType);
        TMaybe<ELossFunction> lossFunction = Nothing();
        if (!lossFunctionName.empty()) {
            lossFunction = FromString<ELossFunction>(lossFunctionName);
        }
        bool isRMSEWithUncertainty = lossFunction == ELossFunction::RMSEWithUncertainty;
        if (isUncertainty) {
            ui32 predictionDim = isRMSEWithUncertainty ? 1 : approxDimension / ensemblesCount;
            TVector<TString> uncertaintyHeaders;
            if (predictionType == EPredictionType::VirtEnsembles) {
                uncertaintyHeaders = {"ApproxRawFormulaVal"};
                if (isRMSEWithUncertainty) {
                    uncertaintyHeaders.push_back("Var");
                }
            } else {
                if (IsRegressionMetric(*lossFunction)) {
                    uncertaintyHeaders = {"MeanPredictions", "KnowledgeUnc"}; // KnowledgeUncertainty
                    if (isRMSEWithUncertainty) {
                        uncertaintyHeaders.push_back("DataUnc"); // DataUncertainty
                    }
                } else {
                    uncertaintyHeaders = {"DataUnc", "TotalUnc"};
                }
                ensemblesCount = 1;
                predictionDim = 1;
            }
            headers.reserve(ensemblesCount * predictionDim * uncertaintyHeaders.size());
            // TODO(eermishkina): support multiRMSE
            for (ui32 veId = 0; veId < ensemblesCount; ++veId) {
                for (ui32 dimId  = 0; dimId  < predictionDim; ++dimId ) {
                    for (const auto &name: uncertaintyHeaders) {
                        TStringBuilder str;
                        str << predictionType << ":" << name;
                        if (predictionDim > 1) {
                            str << (isMultiTarget ? ":Dim=" : ":Class=")
                                << visibleLabelsHelper.GetVisibleClassNameFromClass(dimId );
                        }
                        if (ensemblesCount > 1) {
                            str << ":vEnsemble=" << veId;
                        }
                        headers.push_back(str);
                    }
                }

            }
            return headers;
        }

        headers.reserve(classCount);
        for (ui32 classId = 0; classId < classCount; ++classId) {
            TStringBuilder str;
            str << predictionType;
            if (classCount > 1) {
                if (lossFunction == ELossFunction::RMSEWithUncertainty) {
                    if (classId == 0) {
                        str << "Mean";
                    } else {
                        str << (predictionType == EPredictionType::RMSEWithUncertainty ? "Std" : "Log(Std)");
                    }
                } else {
                    str << (isMultiTarget ? ":Dim=" : ":Class=")
                        << visibleLabelsHelper.GetVisibleClassNameFromClass(classId);
                }
            }
            if (evalParameters && (evalParameters->first != evalParameters->second)) {
                str << ":TreesCount=[" << startTreeIndex << "," <<
                    Min(startTreeIndex + evalParameters->first, evalParameters->second) << ")";
            }
            headers.push_back(str);
        }
        return headers;
    }

} // namespace NCB
