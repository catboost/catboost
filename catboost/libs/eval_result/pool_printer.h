#pragma once

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data/meta_info.h>
#include <catboost/private/libs/data_util/line_data_reader.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/quantized_pool/pool.h>

#include <util/generic/hash.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/stream/output.h>
#include <util/system/types.h>

#include <typeindex>


namespace NCB {

    class IPoolColumnsPrinter : public TThrRefBase {
    public:
        virtual void OutputColumnByType(IOutputStream* outstream, ui64 docId, EColumn columnType) = 0;
        virtual void OutputFeatureColumnByIndex(IOutputStream* outstream, ui64 docId, ui32 featureId) = 0;
        virtual void OutputAuxiliaryColumn(IOutputStream* outstream, ui64 docId, ui32 auxiliaryColumnId, const TString& columnName) = 0;
        virtual bool ValidAuxiliaryColumn(const TString& columnName) = 0;
        virtual ui32 GetAuxiliaryColumnId(const TString& columnName) = 0;
        virtual void UpdateColumnTypeInfo(const TMaybe<TDataColumnsMetaInfo>& /*columnsMetaInfo*/) {}
        virtual ~IPoolColumnsPrinter() = default;
        virtual std::type_index GetOutputFeatureType(ui32 featureId) = 0;
        bool HasDocIdColumn = false;
    };

    // pass this struct to to IPoolColumnsPrinter constructor
    struct TPoolColumnsPrinterPullArgs {
        TPathWithScheme PoolPath;
        const TDsvFormatOptions Format;
        const TMaybe<TDataColumnsMetaInfo> ColumnsMetaInfo;
    };

    // pass this struct to to IPoolColumnsPrinter constructor
    struct TLineDataPoolColumnsPrinterPushArgs {
        THolder<ILineDataReader> Reader;
        const TDsvFormatOptions Format;
        const TMaybe<TDataColumnsMetaInfo> ColumnsMetaInfo;
    };


    class TDSVPoolColumnsPrinter : public IPoolColumnsPrinter {
    public:
        TDSVPoolColumnsPrinter(TPoolColumnsPrinterPullArgs&& args);
        TDSVPoolColumnsPrinter(TLineDataPoolColumnsPrinterPushArgs&& args);
        void OutputColumnByType(IOutputStream* outStream, ui64 docId, EColumn columnType) override;
        void OutputFeatureColumnByIndex(IOutputStream* outStream, ui64 docId, ui32 featureId) override;
        void OutputAuxiliaryColumn(IOutputStream* outStream, ui64 docId, ui32 auxiliaryColumnId, const TString& columnName) override;
        bool ValidAuxiliaryColumn(const TString& columnName) override;
        ui32 GetAuxiliaryColumnId(const TString& columnName) override;
        void UpdateColumnTypeInfo(const TMaybe<TDataColumnsMetaInfo>& columnsMetaInfo) override;
        std::type_index GetOutputFeatureType(ui32 featureId) override;

    private:
        const TString& GetCell(ui64 docId, ui32 colId);

        THolder<ILineDataReader> LineDataReader;
        char Delimiter;
        ui64 DocId;
        TVector<TString> Columns;
        THashMap<EColumn, ui32> FromColumnTypeToColumnId;
        TVector<ui32> FromExternalIdToColumnId;
        THashMap<TString, ui32> AuxiliaryColumnNameToId;
        TMaybe<TDataColumnsMetaInfo> ColumnsMetaInfo;
    };

    class TQuantizedPoolColumnsPrinter : public IPoolColumnsPrinter {
    public:
        TQuantizedPoolColumnsPrinter(TPoolColumnsPrinterPullArgs&& args);
        void OutputColumnByType(IOutputStream* outStream, ui64 docId, EColumn columnType) override;
        void OutputFeatureColumnByIndex(IOutputStream* outStream, ui64 docId, ui32 featureId) override;
        void OutputAuxiliaryColumn(IOutputStream* outStream, ui64 docId, ui32 auxiliaryColumnId, const TString& columnName) override;
        bool ValidAuxiliaryColumn(const TString& columnName) override;
        ui32 GetAuxiliaryColumnId(const TString& columnName) override;
        std::type_index GetOutputFeatureType(ui32 featureId) override;

    private:
        struct ColumnInfo {
            ui32 LocalColumnIndex = 0;
            ui32 CurrentChunkIndex = 0;
            ui32 CurrentOffset = 0;
            ui64 CurrentDocId = 0;
            TString CurrentToken = "";
            TVector<ui32> CorrectChunkOrder;
        };

        const TString GetStringColumnToken(ui64 docId, EColumn columnType);
        const TString GetFloatColumnToken(ui64 docId, EColumn columnType);

        TQuantizedPool QuantizedPool;
        THashMap<EColumn, ColumnInfo> ColumnsInfo;
    };

    using TPoolColumnsPrinterLoaderFactory =
    NObjectFactory::TParametrizedObjectFactory<IPoolColumnsPrinter,
        TString,
        TPoolColumnsPrinterPullArgs>;

} // namespace NCB
