#pragma once

#include "eval_helpers.h"
#include "pool_printer.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data/weights.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/labels/external_label_helper.h>
#include <catboost/private/libs/options/enums.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/output.h>
#include <util/system/compiler.h>
#include <util/system/types.h>

#include <utility>
#include <typeindex>

namespace NCB {

    class IColumnPrinter {
    public:
        virtual ~IColumnPrinter() = default;
        virtual void OutputValue(IOutputStream* outstream, size_t docIndex) = 0;
        virtual void OutputHeader(IOutputStream* outstream) = 0;
        virtual void GetValue(size_t docIndex, TColumnPrinterOuputType* result) = 0;
        virtual TString GetAfterColumnDelimiter() const {
            return "\t";
        }
        virtual std::type_index GetOutputType() = 0;
    };


    template <typename T>
    class TArrayPrinter: public IColumnPrinter {
    public:
        TArrayPrinter(TConstArrayRef<T> array, const TString& header)
            : Array(NCB::TMaybeOwningConstArrayHolder<T>::CreateNonOwning(array))
            , Header(header)
        {
        }

        TArrayPrinter(TVector<T>&& array, const TString& header)
            : Array(NCB::TMaybeOwningConstArrayHolder<T>::CreateOwning(std::move(array)))
            , Header(header)
        {
        }

        TArrayPrinter(TMaybeOwningConstArrayHolder<T>&& array, const TString& header)
            : Array(std::move(array))
            , Header(header)
        {
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            *outStream << (*Array)[docIndex];
        }

        void GetValue(size_t docIndex, TColumnPrinterOuputType* result) override {
            *result = (*Array)[docIndex];
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

        std::type_index GetOutputType() override {
            return typeid(T);
        }

    private:
        const NCB::TMaybeOwningConstArrayHolder<T> Array;
        const TString Header;
    };

    class TWeightsPrinter: public IColumnPrinter {
    public:
        TWeightsPrinter(const TWeights<float>& weights, const TString& header)
            : Weights(weights)
            , Header(header)
        {
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            *outStream << Weights[docIndex];
        }

        void GetValue(size_t docIndex, TColumnPrinterOuputType* result) override {
            *result = Weights[docIndex];
        }

        std::type_index GetOutputType() override {
            return typeid(float);
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

    private:
        const TWeights<float>& Weights;
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

        void GetValue(size_t docIndex, TColumnPrinterOuputType* result) override {
            Y_UNUSED(docIndex);
            *result = Prefix;
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

        std::type_index GetOutputType() override {
            return typeid(T);
        }

        TString GetAfterColumnDelimiter() const override {
            return Delimiter;
        }

    private:
        const TString Prefix;
        const TString Header;
        const TString Delimiter;
    };



    class TFeatureColumnPrinter: public IColumnPrinter {
    public:
        TFeatureColumnPrinter(
            TIntrusivePtr<IPoolColumnsPrinter> printerPtr,
            int featureId,
            TString columnName,
            ui64 docIdOffset
        )
            : PrinterPtr(printerPtr)
            , FeatureId(featureId)
            , ColumnName(std::move(columnName))
            , DocIdOffset(docIdOffset)
        {
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override  {
            PrinterPtr->OutputFeatureColumnByIndex(outStream, DocIdOffset + docIndex, FeatureId);
        }

        void GetValue(size_t docIndex, TColumnPrinterOuputType* result) override {
            TStringStream value;
            PrinterPtr->OutputFeatureColumnByIndex(&value, DocIdOffset + docIndex, FeatureId);
            *result = value.Str();
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << ColumnName;
        }

        std::type_index GetOutputType() override {
            return typeid(TString);
        }

    private:
        TIntrusivePtr<IPoolColumnsPrinter> PrinterPtr;
        int FeatureId;
        TString ColumnName;
        ui64 DocIdOffset;
    };



    class TCatFeaturePrinter: public IColumnPrinter {
    public:
        TCatFeaturePrinter(
            TMaybeOwningArrayHolder<ui32>&& hashedValues,
            const THashMap<ui32, TString>& hashToString,
            const TString& header
        )
            : HashedValues(std::move(hashedValues))
            , HashToString(hashToString)
            , Header(header)
        {
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            *outStream << HashToString.at(HashedValues[docIndex]);
        }

        void GetValue(size_t docIndex, TColumnPrinterOuputType* result) override {
            *result = HashToString.at(HashedValues[docIndex]);
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }

        std::type_index GetOutputType() override {
            return typeid(TString);
        }

    private:
        const TMaybeOwningArrayHolder<ui32> HashedValues;
        const THashMap<ui32, TString>& HashToString;
        const TString Header;
    };



    class TEvalPrinter: public IColumnPrinter {
    public:
        TEvalPrinter(
            const EPredictionType predictionType,
            const TString& header,
            const TVector<double>& approx,
            const TExternalLabelsHelper& visibleLabelsHelper
        )
            : PredictionType(predictionType)
            , Header(header)
            , Approx(approx)
            , VisibleLabelsHelper(visibleLabelsHelper)
        {}

        void OutputValue(IOutputStream* outStream, size_t docIndex) override {
            if (PredictionType == EPredictionType::Class) {
                *outStream << VisibleLabelsHelper.GetVisibleClassNameFromClass(
                    static_cast<int>(Approx[docIndex])
                );
            } else {
                *outStream << Approx[docIndex];
            }
        }

        void GetValue(size_t docIndex, TColumnPrinterOuputType* result) override {
            if (PredictionType == EPredictionType::Class) {
                *result = VisibleLabelsHelper.GetVisibleClassNameFromClass(static_cast<int>(Approx[docIndex]));
            } else {
                *result =  Approx[docIndex];
            }
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << Header;
        }
        std::type_index GetOutputType() override {
            if (PredictionType == EPredictionType::Class) {
                return typeid(TString);
            } else {
                return typeid(double);
            }
        }

    private:
        EPredictionType PredictionType;
        TString Header;
        TVector<double> Approx;
        TExternalLabelsHelper VisibleLabelsHelper;
    };

    void PushBackEvalPrinters(
        const TVector<TVector<TVector<double>>>& rawValues,
        const EPredictionType predictionType,
        const TString& lossFunctionName,
        const TMaybe<TString>& modelName,
        bool isMultiTarget,
        size_t ensemblesCount,
        const TExternalLabelsHelper& visibleLabelsHelper,
        TMaybe<std::pair<size_t, size_t>> evalParameters,
        TVector<THolder<IColumnPrinter>>* result,
        NPar::ILocalExecutor* executor,
        double binClassLogitThreshold
    );


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

        void GetValue(size_t docIndex, TColumnPrinterOuputType* result) override {
            TStringStream value;
            OutputValue(&value, docIndex);
            if (GetOutputType() == typeid(double)) {
                *result = FromString<double>(value.Str());
            } else {
                *result = value.Str();
            }
        }

        std::type_index GetOutputType() override {
            switch (ColumnType) {
                case EColumn::Label:
                case EColumn::Weight:
                case EColumn::GroupWeight:
                    return typeid(double);
                case EColumn::SampleId:
                case EColumn::GroupId:
                case EColumn::SubgroupId:
                    return typeid(TString);
                default:
                    CB_ENSURE(false, "Unknown output columnType");
            }
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
                PrinterPtr->OutputColumnByType(outStream, DocIdOffset + docIndex, EColumn::SampleId);
            }
        }

        void GetValue(size_t docIndex, TColumnPrinterOuputType* result) override {
            if (NeedToGenerate) {
                *result = DocIdOffset + docIndex;
            } else {
                TStringStream value;
                OutputValue(&value, docIndex);
                *result = value.Str();
            }
        }

        std::type_index GetOutputType() override {
            return NeedToGenerate ? typeid(ui64) : typeid(TString);
        }

        bool NeedPrinterPtr() {
            return !NeedToGenerate;
        }

    private:
        TIntrusivePtr<IPoolColumnsPrinter> PrinterPtr;
        bool NeedToGenerate;
        ui64 DocIdOffset;
        TString Header;
    };



    class TAuxiliaryColumnPrinter: public IColumnPrinter {
    public:
        TAuxiliaryColumnPrinter(
            TIntrusivePtr<IPoolColumnsPrinter> printerPtr,
            TString columnName,
            ui64 docIdOffset
        )
            : PrinterPtr(printerPtr)
            , AuxiliaryColumnId(printerPtr->GetAuxiliaryColumnId(columnName))
            , ColumnName(std::move(columnName))
            , DocIdOffset(docIdOffset)
        {
        }

        void OutputValue(IOutputStream* outStream, size_t docIndex) override  {
            PrinterPtr->OutputAuxiliaryColumn(outStream, DocIdOffset + docIndex, AuxiliaryColumnId, ColumnName);
        }

        void GetValue(size_t docIndex, TColumnPrinterOuputType* result) override {
            TStringStream value;
            PrinterPtr->OutputAuxiliaryColumn(&value, DocIdOffset + docIndex, AuxiliaryColumnId, ColumnName);
            *result = value.Str();
        }

        void OutputHeader(IOutputStream* outStream) override {
            *outStream << ColumnName;
        }

        std::type_index GetOutputType() override {
            return typeid(TString);
        }

    private:
        TIntrusivePtr<IPoolColumnsPrinter> PrinterPtr;
        ui32 AuxiliaryColumnId;
        TString ColumnName;
        ui64 DocIdOffset;
    };

    TVector<TString> CreatePredictionTypeHeader(
        ui32 approxDimension,
        bool isMultiTarget,
        EPredictionType predictionType,
        const TExternalLabelsHelper& visibleLabelsHelper,
        const TString& lossFunctionName,
        const TMaybe<TString>& modelName,
        size_t ensemblesCount,
        ui32 startTreeIndex = 0,
        std::pair<size_t, size_t>* evalParameters = nullptr
    );
} // namespace NCB
