#pragma once

#include <catboost/libs/data_util/line_data_reader.h>
#include <catboost/libs/quantized_pool/pool.h>
#include <catboost/libs/quantized_pool/serialization.h>
#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>

#include <util/generic/hash_set.h>
#include <util/generic/xrange.h>
#include <util/system/byteorder.h>
#include <util/system/unaligned_mem.h>


namespace NCB {

    class IPoolColumnsPrinter : public TThrRefBase {
    public:
        virtual void OutputColumnByType(IOutputStream* outstream, ui64 docId, EColumn columnType) = 0;
        virtual void OutputColumnByIndex(IOutputStream* outstream, ui64 docId, ui32 columnId) = 0;
        virtual ~IPoolColumnsPrinter() = default;
        bool HasDocIdColumn = false;
    };

    class TDSVPoolColumnsPrinter : public IPoolColumnsPrinter {
    public:
        TDSVPoolColumnsPrinter(
            const TPathWithScheme& testSetPath,
            const TDsvFormatOptions& format,
            const TMaybe<TPoolColumnsMetaInfo>& columnsMetaInfo
        );
        void OutputColumnByType(IOutputStream* outStream, ui64 docId, EColumn columnType) override;
        void OutputColumnByIndex(IOutputStream* outStream, ui64 docId, ui32 columnId) override;

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
        };

        const TString GetStringColumnToken(ui64 docId, EColumn columnType);
        const TString GetFloatColumnToken(ui64 docId, EColumn columnType);

        TQuantizedPool QuantizedPool;
        THashMap<EColumn, ColumnInfo> ColumnsInfo;
    };

} // namespace NCB
