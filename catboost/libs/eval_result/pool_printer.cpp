#include "pool_printer.h"

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/quantized_pool/serialization.h>

#include <util/generic/xrange.h>
#include <util/string/cast.h>
#include <util/string/iterator.h>
#include <util/system/byteorder.h>
#include <util/system/unaligned_mem.h>


namespace NCB {

    TDSVPoolColumnsPrinter::TDSVPoolColumnsPrinter(
        const TPathWithScheme& testSetPath,
        const TDsvFormatOptions& format,
        const TMaybe<TDataColumnsMetaInfo>& columnsMetaInfo
    )
        : LineDataReader(GetLineDataReader(testSetPath, format))
        , Delimiter(format.Delimiter)
        , DocId(-1)
    {
        UpdateColumnTypeInfo(columnsMetaInfo);
    }

    void TDSVPoolColumnsPrinter::OutputColumnByType(IOutputStream* outStream, ui64 docId, EColumn columnType) {
        CB_ENSURE(FromColumnTypeToColumnId.contains(columnType),
            "You can not output " << ToString(columnType) << " column by type");
        *outStream << GetCell(docId, FromColumnTypeToColumnId[columnType]);
    }

    void TDSVPoolColumnsPrinter::OutputColumnByIndex(IOutputStream* outStream, ui64 docId, ui32 columnId) {
        *outStream << GetCell(docId, columnId);
    }

    void TDSVPoolColumnsPrinter::UpdateColumnTypeInfo(const TMaybe<TDataColumnsMetaInfo>& columnsMetaInfo) {
        if (columnsMetaInfo.Defined()) {
            for (ui32 columnId : xrange(columnsMetaInfo->Columns.size())) {
                const auto columnType = columnsMetaInfo->Columns[columnId].Type;
                FromColumnTypeToColumnId[columnType] = columnId;
                if (columnType == EColumn::SampleId) {
                    HasDocIdColumn = true;
                }
            }
        }
    }

    const TString& TDSVPoolColumnsPrinter::GetCell(ui64 docId, ui32 colId) {
        if (docId == DocId + 1) {
            ++DocId;
            TString line;
            CB_ENSURE(LineDataReader->ReadLine(&line),
                      "there's no line in pool for " << DocId);
            Columns.clear();
            for (const auto& typeName : StringSplitter(line).Split(Delimiter)) {
                Columns.push_back(FromString<TString>(typeName.Token()));
            }
        }
        CB_ENSURE(docId == DocId, "only serial lines possible to output");
        return Columns[colId];
    }

    TQuantizedPoolColumnsPrinter::TQuantizedPoolColumnsPrinter(const TPathWithScheme& testSetPath)
        : QuantizedPool(LoadQuantizedPool(testSetPath.Path, {/*LockMemory=*/false, /*Precharge=*/false}))
    {
        for (const ui32 columnId : xrange(QuantizedPool.ColumnTypes.size())) {
            const auto columnType = QuantizedPool.ColumnTypes[columnId];
            ui32 localColumnIndex;
            switch (columnType) {
                case EColumn::SampleId:
                    HasDocIdColumn = true;
                    localColumnIndex = QuantizedPool.StringDocIdLocalIndex;
                    break;
                case EColumn::GroupId:
                    localColumnIndex = QuantizedPool.StringGroupIdLocalIndex;
                    break;
                case EColumn::SubgroupId:
                    localColumnIndex = QuantizedPool.StringSubgroupIdLocalIndex;
                    break;
                default:
                    localColumnIndex = columnId;
                    break;
            }
            ColumnsInfo[columnType].LocalColumnIndex = localColumnIndex;
        }
    }

    void TQuantizedPoolColumnsPrinter::OutputColumnByType(IOutputStream* outStream, ui64 docId, EColumn columnType) {
        CB_ENSURE(ColumnsInfo.contains(columnType),
            "Pool doesn't have " << ToString(columnType) << " column.");

        TString token;
        switch (columnType) {
            case EColumn::Label:
            case EColumn::Weight:
            case EColumn::GroupWeight:
                token = GetFloatColumnToken(docId, columnType);
                break;
            case EColumn::SampleId:
            case EColumn::GroupId:
            case EColumn::SubgroupId:
                token = GetStringColumnToken(docId, columnType);
            default:
                CB_ENSURE("Unsupported output columnType for Quantized pool.");
        }

        *outStream << token;
    }

    void TQuantizedPoolColumnsPrinter::OutputColumnByIndex(IOutputStream* /*outStream*/, ui64 /*docId*/, ui32 /*columnId*/) {
        CB_ENSURE(false, "Not Implemented for Quantized Pools");
    }

    const TString TQuantizedPoolColumnsPrinter::GetStringColumnToken(ui64 docId, EColumn columnType) {
        CB_ENSURE(QuantizedPool.HasStringColumns);
        auto& columnInfo = ColumnsInfo[columnType];

        if (docId == columnInfo.CurrentDocId - 1) {
            return columnInfo.CurrentToken;
        }

        CB_ENSURE(columnInfo.CurrentDocId == docId, "Only serial lines possible to output.");
        const auto& chunks = QuantizedPool.Chunks[columnInfo.LocalColumnIndex];
        const auto& chunk = chunks[columnInfo.CurrentChunkIndex];

        CB_ENSURE(chunk.Chunk->Quants()->size() > columnInfo.CurrentOffset);
        const ui8* data = chunk.Chunk->Quants()->data();

        CB_ENSURE(chunk.Chunk->Quants()->size() - columnInfo.CurrentOffset >= sizeof(ui32));
        const ui32 tokenSize = LittleToHost(ReadUnaligned<ui32>(data + columnInfo.CurrentOffset));
        columnInfo.CurrentOffset += sizeof(ui32);

        CB_ENSURE(chunk.Chunk->Quants()->size() - columnInfo.CurrentOffset >= tokenSize);
        columnInfo.CurrentToken = TString(reinterpret_cast<const char*>(data + columnInfo.CurrentOffset), tokenSize);
        columnInfo.CurrentOffset += tokenSize;
        ++columnInfo.CurrentDocId;

        if (chunk.Chunk->Quants()->size() == columnInfo.CurrentOffset) {
            columnInfo.CurrentOffset = 0;
            ++columnInfo.CurrentChunkIndex;
        }
        return columnInfo.CurrentToken;
    }

    const TString TQuantizedPoolColumnsPrinter::GetFloatColumnToken(ui64 docId, EColumn columnType) {
        auto& columnInfo = ColumnsInfo[columnType];

        if (docId == columnInfo.CurrentDocId - 1) {
            return columnInfo.CurrentToken;
        }

        CB_ENSURE(columnInfo.CurrentDocId == docId, "Only serial lines possible to output.");
        const auto& chunks = QuantizedPool.Chunks[columnInfo.LocalColumnIndex];
        const auto& chunk = chunks[columnInfo.CurrentChunkIndex];

        CB_ENSURE(chunk.Chunk->Quants()->size() > columnInfo.CurrentOffset);
        const ui8* data = chunk.Chunk->Quants()->data();

        CB_ENSURE(chunk.Chunk->Quants()->size() - columnInfo.CurrentOffset >= sizeof(float));
        columnInfo.CurrentToken = ToString(ReadUnaligned<float>(data + columnInfo.CurrentOffset));
        columnInfo.CurrentOffset += sizeof(float);
        ++columnInfo.CurrentDocId;

        if (chunk.Chunk->Quants()->size() == columnInfo.CurrentOffset) {
            columnInfo.CurrentOffset = 0;
            ++columnInfo.CurrentChunkIndex;
        }
        return columnInfo.CurrentToken;
    }

} // namespace NCB
