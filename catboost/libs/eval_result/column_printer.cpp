#include "column_printer.h"


namespace NCB {

// TEvalPrinter

    TEvalPrinter::TEvalPrinter(
        NPar::TLocalExecutor* executor,
        const TVector<TVector<TVector<double>>>& rawValues,
        const EPredictionType predictionType,
        const TExternalLabelsHelper& visibleLabelsHelper,
        TMaybe<std::pair<size_t, size_t>> evalParameters)
        : VisibleLabelsHelper(visibleLabelsHelper) {
        int begin = 0;
        for (const auto& raws : rawValues) {
            CB_ENSURE(VisibleLabelsHelper.IsInitialized() == IsMulticlass(raws),
                      "Inappropriated usage of visible label helper: it MUST be initialized ONLY for multiclass problem");
            const auto& approx = VisibleLabelsHelper.IsInitialized() ? MakeExternalApprox(raws, VisibleLabelsHelper) : raws;
            Approxes.push_back(PrepareEval(predictionType, approx, executor));
            for (int classId = 0; classId < Approxes.back().ysize(); ++classId) {
                TStringBuilder str;
                str << predictionType;
                if (Approxes.back().ysize() > 1) {
                    str << ":Class=" << VisibleLabelsHelper.GetVisibleClassNameFromClass(classId);
                }
                if (rawValues.ysize() > 1) {
                    str << ":TreesCount=[" << begin << "," << Min(begin + evalParameters->first, evalParameters->second) << ")";
                }
                Header.push_back(str);
            }
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

} // namespace NCB
