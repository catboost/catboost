#include "column_printer.h"

#include "eval_helpers.h"

#include <util/generic/utility.h>
#include <util/string/builder.h>


namespace NCB {

// TEvalPrinter

    TEvalPrinter::TEvalPrinter(
        NPar::TLocalExecutor* executor,
        const TVector<TVector<TVector<double>>>& rawValues,
        const EPredictionType predictionType,
        const TString& lossFunctionName,
        const TExternalLabelsHelper& visibleLabelsHelper,
        TMaybe<std::pair<size_t, size_t>> evalParameters)
        : VisibleLabelsHelper(visibleLabelsHelper) {
        int begin = 0;
        for (const auto& raws : rawValues) {
            CB_ENSURE(VisibleLabelsHelper.IsInitialized() == IsMulticlass(raws),
                      "Inappropriate usage of visible label helper: it MUST be initialized ONLY for multiclass problem");
            const auto& approx = VisibleLabelsHelper.IsInitialized() ? MakeExternalApprox(raws, VisibleLabelsHelper) : raws;
            Approxes.push_back(PrepareEval(predictionType, lossFunctionName, approx, executor));
            const auto& headers = CreatePredictionTypeHeader(approx.size(), predictionType, VisibleLabelsHelper, begin, evalParameters.Get());
            Header.insert(Header.end(), headers.begin(), headers.end());
            if (evalParameters) {
                begin += evalParameters->first;
            }
        }
    }

    void TEvalPrinter::OutputValue(IOutputStream* outStream, size_t docIndex) {
        TString delimiter = "";
        if (VisibleLabelsHelper.IsInitialized() && Approxes.back().ysize() == 1) { // class labels
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
                str << ":Class=" << visibleLabelsHelper.GetVisibleClassNameFromClass(classId);
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
