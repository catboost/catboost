#pragma once

#include "pool_printer.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/labels/external_label_helper.h>
#include <catboost/libs/options/enums.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/maybe.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/output.h>
#include <util/system/compiler.h>
#include <util/system/types.h>

#include <utility>


namespace NCB {

    class IColumnPrinter {
    public:
        virtual void OutputValue(IOutputStream* outstream, size_t docIndex) = 0;
        virtual void OutputHeader(IOutputStream* outstream) = 0;
        virtual TString GetAfterColumnDelimiter() const {
            return "\t";
        }
        virtual ~IColumnPrinter() = default;
    };



    template <typename T>
    class TVectorPrinter: public IColumnPrinter {
    public:
        TVectorPrinter(const TVector<T>& targetRef, const TString& header)
            : Ref(targetRef)
            , Header(header)
        {
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            *outStream << Ref[docIndex];
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

    private:
        const TVector<T>& Ref;
        const TString Header;
    };



    template <typename T>
    class TPrefixPrinter: public IColumnPrinter {
    public:
        TPrefixPrinter(const TString& prefix, const TString& header, const TString& delimiter)
            : Prefix(prefix)
            , Header(header)
            , Delimiter(delimiter)
        {
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            Y_UNUSED(docIndex);
            *outStream << Prefix;
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

        TString GetAfterColumnDelimiter() const override {
            return Delimiter;
        }

    private:
        const TString Prefix;
        const TString Header;
        const TString Delimiter;
    };



    class TNumColumnPrinter: public IColumnPrinter {
    public:
        TNumColumnPrinter(TIntrusivePtr<IPoolColumnsPrinter> printerPtr, int colId, ui64 docIdOffset)
            : PrinterPtr(printerPtr)
            , ColId(colId)
            , DocIdOffset(docIdOffset)
        {
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override  {
            PrinterPtr->OutputColumnByIndex(outStream, DocIdOffset + docIndex, ColId);
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << '#' << ColId;
        }

    private:
        TIntrusivePtr<IPoolColumnsPrinter> PrinterPtr;
        int ColId;
        ui64 DocIdOffset;
    };



    class TCatFeaturePrinter: public IColumnPrinter {
    public:
        TCatFeaturePrinter(const TVector<float>& hashedValues,
                           const THashMap<int, TString>& hashToString,
                           const TString& header)
            : HashedValues(hashedValues)
            , HashToString(hashToString)
            , Header(header)
        {
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            *outStream << HashToString.at(HashedValues[docIndex]);
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

    private:
        const TVector<float>& HashedValues;
        const THashMap<int, TString>& HashToString;
        const TString Header;
    };



    class TEvalPrinter: public IColumnPrinter {
    public:
        TEvalPrinter(
            NPar::TLocalExecutor* executor,
            const TVector<TVector<TVector<double>>>& rawValues,
            const EPredictionType predictionType,
            const TExternalLabelsHelper& visibleLabelsHelper,
            TMaybe<std::pair<size_t, size_t>> evalParameters = TMaybe<std::pair<size_t, size_t>>());
        void OutputValue(IOutputStream* outStream, size_t docIndex) override;
        void OutputHeader(IOutputStream* outStream) override;

    private:
        TVector<TString> Header;
        TVector<TVector<TVector<double>>> Approxes;
        const TExternalLabelsHelper& VisibleLabelsHelper;
    };



    class TColumnPrinter: public IColumnPrinter {
    public:
        TColumnPrinter(
            TIntrusivePtr<IPoolColumnsPrinter> printerPtr,
            EColumn columnType,
            ui64 docIdOffset,
            const TString& header
        )
            : PrinterPtr(printerPtr)
            , ColumnType(columnType)
            , DocIdOffset(docIdOffset)
            , Header(header)
        {
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            CB_ENSURE(PrinterPtr, "It is imposible to output column without Pool.");
            PrinterPtr->OutputColumnByType(outStream, DocIdOffset + docIndex, ColumnType);
        }

    private:
        TIntrusivePtr<IPoolColumnsPrinter> PrinterPtr;
        EColumn ColumnType;
        ui64 DocIdOffset;
        TString Header;
    };



    class TDocIdPrinter: public IColumnPrinter {
    public:
        TDocIdPrinter(TIntrusivePtr<IPoolColumnsPrinter> printerPtr, ui64 docIdOffset, const TString& header)
            : PrinterPtr(printerPtr)
            , NeedToGenerate(!printerPtr || !printerPtr->HasDocIdColumn)
            , DocIdOffset(docIdOffset)
            , Header(header)
        {
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            if (NeedToGenerate) {
                *outStream << DocIdOffset + docIndex;
            } else {
                PrinterPtr->OutputColumnByType(outStream, DocIdOffset + docIndex, EColumn::DocId);
            }
        }

    private:
        TIntrusivePtr<IPoolColumnsPrinter> PrinterPtr;
        bool NeedToGenerate;
        ui64 DocIdOffset;
        TString Header;
    };

    TVector<TString> CreatePredictionTypeHeader(
        ui32 approxDimension,
        EPredictionType predictionType,
        const TExternalLabelsHelper& visibleLabelsHelper,
        ui32 startTreeIndex = 0,
        std::pair<size_t, size_t>* evalParameters = nullptr);
} // namespace NCB
