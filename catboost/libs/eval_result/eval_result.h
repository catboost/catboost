#pragma once

#include "eval_helpers.h"
#include "column_printer.h"


namespace NCB {

    class TEvalResult {
    public:
        TEvalResult() {
            RawValues.resize(1);
        }

        TVector<TVector<TVector<double>>>& GetRawValuesRef();
        const TVector<TVector<TVector<double>>>& GetRawValuesConstRef() const;
        void ClearRawValues();

        /// *Move* data from `rawValues` to `RawValues[0]`
        void SetRawValuesByMove(TVector<TVector<double>>& rawValues);

    private:
        TVector<TVector<TVector<double>>> RawValues; // [evalIter][dim][docIdx]
    };

    void ValidateColumnOutput(
        const TVector<TString>& outputColumns,
        const TPool& pool,
        bool isPartOfFullTestSet=false,
        bool CV_mode=false);

    TIntrusivePtr<IPoolColumnsPrinter> CreatePoolColumnPrinter(
        const TPathWithScheme& testSetPath,
        const TDsvFormatOptions& testSetFormat,
        const TMaybe<TPoolColumnsMetaInfo>& columnsMetaInfo = {}
    );

    void OutputEvalResultToFile(
        const TEvalResult& evalResult,
        NPar::TLocalExecutor* executor,
        const TVector<TString>& outputColumns,
        const TExternalLabelsHelper& visibleLabelsHelper,
        const TPool& pool,
        bool isPartOfTestSet, // pool is a part of test set, can't output testSetPath columns
        IOutputStream* outputStream,
        TIntrusivePtr<IPoolColumnsPrinter> poolColumnsPrinter,
        std::pair<int, int> testFileWhichOf,
        bool writeHeader = true,
        ui64 docIdOffset = 0,
        TMaybe<std::pair<size_t, size_t>> evalParameters = TMaybe<std::pair<size_t, size_t>>());

    void OutputEvalResultToFile(
        const TEvalResult& evalResult,
        int threadCount,
        const TVector<TString>& outputColumns,
        const TExternalLabelsHelper& visibleLabelsHelper,
        const TPool& pool,
        bool isPartOfTestSet, // pool is a part of test set, can't output testSetPath columns
        IOutputStream* outputStream,
        const NCB::TPathWithScheme& testSetPath,
        std::pair<int, int> testFileWhichOf,
        const NCB::TDsvFormatOptions& testSetFormat,
        bool writeHeader = true,
        ui64 docIdOffset = 0);

    void OutputGpuEvalResultToFile(
        const TVector<TVector<double>>& approxes,
        ui32 threadCount,
        TConstArrayRef<TString> outputColumns,
        const TPathWithScheme& testSetPath,
        const TDsvFormatOptions& testSetFormat,
        const TPoolMetaInfo& poolMetaInfo,
        const TString& serializedMulticlassLabelParams,
        const TString& evalOutputFileName);

} // namespace NCB
