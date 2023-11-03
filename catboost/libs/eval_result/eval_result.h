#pragma once

#include "column_printer.h"
#include "pool_printer.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/private/libs/data_util/line_data_reader.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/labels/external_label_helper.h>

#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/output.h>
#include <util/system/types.h>

#include <utility>


namespace NCB {

    class TEvalResult {
    public:
        TEvalResult(size_t ensemblesCount = 1)
            : EnsemblesCount(ensemblesCount) {
            RawValues.resize(1);
        }

        TVector<TVector<TVector<double>>>& GetRawValuesRef();
        const TVector<TVector<TVector<double>>>& GetRawValuesConstRef() const;
        void ClearRawValues();

        /// *Move* data from `rawValues` to `RawValues[0]`
        void SetRawValuesByMove(TVector<TVector<double>>& rawValues);
        size_t GetEnsemblesCount() const;

    private:
        size_t EnsemblesCount;
        TVector<TVector<TVector<double>>> RawValues; // [evalIter][dim][docIdx]
    };

    void ValidateColumnOutput(
        const TVector<TString>& outputColumns,
        const TDataProvider& pool,
        bool cvMode=false);

    TIntrusivePtr<IPoolColumnsPrinter> CreatePoolColumnPrinter(
        const TPathWithScheme& testSetPath,
        const TDsvFormatOptions& testSetFormat,
        const TMaybe<TDataColumnsMetaInfo>& columnsMetaInfo = {}
    );

    TVector<THolder<IColumnPrinter>> InitializeColumnWriter(
        const TEvalResult& evalResult,
        NPar::ILocalExecutor* executor,
        const TVector<TString>& outputColumns,
        const TString& lossFunctionName,
        const TExternalLabelsHelper& visibleLabelsHelper,
        const TDataProvider& pool,
        TIntrusivePtr<IPoolColumnsPrinter> poolColumnsPrinter,
        std::pair<int, int> testFileWhichOf,
        ui64 docIdOffset,
        bool* needColumnsPrinterPtr,
        TMaybe<std::pair<size_t, size_t>> evalParameters = TMaybe<std::pair<size_t, size_t>>(),
        double binClassLogitThreshold = DEFAULT_BINCLASS_LOGIT_THRESHOLD);

    void OutputEvalResultToFile(
        const TEvalResult& evalResult,
        NPar::ILocalExecutor* executor,
        const TVector<TString>& outputColumns,
        const TString& lossFunctionName,
        const TExternalLabelsHelper& visibleLabelsHelper,
        const TDataProvider& pool,
        IOutputStream* outputStream,
        TIntrusivePtr<IPoolColumnsPrinter> poolColumnsPrinter,
        std::pair<int, int> testFileWhichOf,
        bool writeHeader = true,
        ui64 docIdOffset = 0,
        TMaybe<std::pair<size_t, size_t>> evalParameters = TMaybe<std::pair<size_t, size_t>>(),
        double binClassLogitThreshold = DEFAULT_BINCLASS_LOGIT_THRESHOLD);

    void OutputEvalResultToFile(
        const TEvalResult& evalResult,
        NPar::ILocalExecutor* executor,
        const TVector<TString>& outputColumns,
        const TString& lossFunctionName,
        const TExternalLabelsHelper& visibleLabelsHelper,
        const TDataProvider& pool,
        IOutputStream* outputStream,
        const NCB::TPathWithScheme& testSetPath,
        std::pair<int, int> testFileWhichOf,
        const NCB::TDsvFormatOptions& testSetFormat,
        bool writeHeader = true,
        ui64 docIdOffset = 0,
        double binClassLogitThreshold = DEFAULT_BINCLASS_LOGIT_THRESHOLD);

} // namespace NCB
