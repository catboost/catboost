#include "column_printer.h"

#include "eval_helpers.h"

#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/loss_description.h>
#include <util/generic/utility.h>
#include <util/string/builder.h>


namespace NCB {

// TEvalPrinter

    TEvalPrinter::TEvalPrinter(
        NPar::TLocalExecutor* executor,
        const TVector<TVector<TVector<double>>>& rawValues,
        const EPredictionType predictionType,
        const TString& lossFunctionName,
        ui32 targetDimension,
        const TExternalLabelsHelper& visibleLabelsHelper,
        TMaybe<std::pair<size_t, size_t>> evalParameters)
        : PredictionType(predictionType)
        ,VisibleLabelsHelper(visibleLabelsHelper) {
        int begin = 0;
        const bool isMultiTarget = targetDimension > 1;
        const bool callMakeExternalApprox
            = VisibleLabelsHelper.IsInitialized()
                && (VisibleLabelsHelper.GetExternalApproxDimension() > 1);
        for (const auto& raws : rawValues) {
            const auto& approx = callMakeExternalApprox ? MakeExternalApprox(raws, VisibleLabelsHelper) : raws;
            Approxes.push_back(PrepareEval(predictionType, lossFunctionName, approx, executor));

            const auto& headers = CreatePredictionTypeHeader(
                approx.size(),
                isMultiTarget,
                predictionType,
                VisibleLabelsHelper,
                begin,
                evalParameters.Get()
            );
            Header.insert(Header.end(), headers.begin(), headers.end());
            if (evalParameters) {
                begin += evalParameters->first;
            }
        }
    }

    void TEvalPrinter::OutputValue(IOutputStream* outStream, size_t docIndex) {
        TString delimiter = "";
        if (PredictionType == EPredictionType::Class) {
            for (const auto& approxes : Approxes) {
                for (const auto& approx : approxes) {
                    *outStream << delimiter
                               << VisibleLabelsHelper.GetVisibleClassNameFromClass(static_cast<int>(approx[docIndex]));
                    delimiter = "\t";
                }
            }
        } else {
            for (const auto& approxes : Approxes) {
                for (const auto& approx : approxes) {
                    *outStream << delimiter << approx[docIndex];
                    delimiter = "\t";
                }
            }
        }
    }

    void TEvalPrinter::OutputHeader(IOutputStream* outStream) {
        for (int idx = 0; idx < Header.ysize(); ++idx) {
            if (idx > 0) {
                *outStream << "\t";
            }
            *outStream << Header[idx];
        }
    }

    TVector<TString> CreatePredictionTypeHeader(
        ui32 approxDimension,
        bool isMultiTarget,
        EPredictionType predictionType,
        const TExternalLabelsHelper& visibleLabelsHelper,
        ui32 startTreeIndex,
        std::pair<size_t, size_t>* evalParameters) {

        const ui32 classCount = (predictionType == EPredictionType::Class) ? 1 : approxDimension;
        TVector<TString> headers;
        headers.reserve(classCount);
        for (ui32 classId = 0; classId < classCount; ++classId) {
            TStringBuilder str;
            str << predictionType;
            if (classCount > 1) {
                str << (isMultiTarget ?  ":Dim=" : ":Class=")  << visibleLabelsHelper.GetVisibleClassNameFromClass(classId);
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
