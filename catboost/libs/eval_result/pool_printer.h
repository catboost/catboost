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


namespace NCB {

    class IPoolColumnsPrinter : public TThrRefBase {
    public:
        virtual void OutputColumnByType(IOutputStream* outstream, ui64 docId, EColumn columnType) = 0;
        virtual void OutputColumnByIndex(IOutputStream* outstream, ui64 docId, ui32 columnId) = 0;
        virtual void UpdateColumnTypeInfo(const TMaybe<TDataColumnsMetaInfo>& /*columnsMetaInfo*/) {}
        virtual ~IPoolColumnsPrinter() = default;
        bool HasDocIdColumn = false;
    private:
        // TODO(nikitxskv): Temporary solution until MLTOOLS-140 is implemented.
        THashMap<EColumn, ui32> FromColumnTypeToColumnId; // Only for DSV pools
    };

    class TDSVPoolColumnsPrinter : public IPoolColumnsPrinter {
    public:
        TDSVPoolColumnsPrinter(
            const TPathWithScheme& testSetPath,
            const TDsvFormatOptions& format,
            const TMaybe<TDataColumnsMetaInfo>& columnsMetaInfo
        );
        void OutputColumnByType(IOutputStream* outStream, ui64 docId, EColumn columnType) override;
        void OutputColumnByIndex(IOutputStream* outStream, ui64 docId, ui32 columnId) override;
        void UpdateColumnTypeInfo(const TMaybe<TDataColumnsMetaInfo>& columnsMetaInfo) override;

    private:
        const TString& GetCell(ui64 docId, ui32 colId);

        THolder<ILineDataReader> LineDataReader;
        char Delimiter;
        ui64 DocId;
        TVector<TString> Columns;
        THashMap<EColumn, ui32> FromColumnTypeToColumnId;
    };

    class TQuantizedPoolColumnsPrinter : public IPoolColumnsPrinter {
    public:
        TQuantizedPoolColumnsPrinter(const TPathWithScheme& testSetPath);
        void OutputColumnByType(IOutputStream* outStream, ui64 docId, EColumn columnType) override;
        void OutputColumnByIndex(IOutputStream* outStream, ui64 docId, ui32 columnId) override;

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

} // namespace NCB
