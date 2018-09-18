#include "pool_printer.h"

#include <util/string/iterator.h>


namespace NCB {

    TDSVPoolColumnsPrinter::TDSVPoolColumnsPrinter(
        const TPathWithScheme& testSetPath,
        const TDsvFormatOptions& format,
        const TVector<TColumn>& columnTypes
    )
        : LineDataReader(GetLineDataReader(testSetPath, format))
        , Delimiter(format.Delimiter)
        , DocId(-1)
    {
        for (ui32 columnId : xrange(columnTypes.size())) {
            const auto columnType = columnTypes[columnId].Type;
            switch (columnType) {
                case EColumn::DocId:
                case EColumn::GroupId:
                case EColumn::SubgroupId:
                    FromColumnTypeToColumnId[columnType] = columnId;
                    break;
                default:
                    break;
            }
        }
    }

    void TDSVPoolColumnsPrinter::OutputColumnByType(IOutputStream* outStream, ui64 docId, EColumn columnType) {
        CB_ENSURE(FromColumnTypeToColumnId.has(columnType),
            "You can not output " << ToString(columnType) << " column by type");
        *outStream << GetCell(docId, FromColumnTypeToColumnId[columnType]);
    }

    void TDSVPoolColumnsPrinter::OutputColumnByIndex(IOutputStream* outStream, ui64 docId, ui32 columnId) {
        *outStream << GetCell(docId, columnId);
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
                case EColumn::DocId:
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
        CB_ENSURE(ColumnsInfo.has(columnType),
            "Pool doesn't have " << ToString(columnType) << " column.");

        TString token;
        switch (columnType) {
            case EColumn::Label:
            case EColumn::Weight:
            case EColumn::GroupWeight:
                token = GetFloatColumnToken(docId, columnType);
                break;
            case EColumn::DocId:
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

        CB_ENSURE(columnInfo.CurrentDocId == docId, "Only serial lines possible to output.");
        const auto& chunks = QuantizedPool.Chunks[columnInfo.LocalColumnIndex];
        const auto& chunk = chunks[columnInfo.CurrentChunkIndex];

        CB_ENSURE(chunk.Chunk->Quants()->size() > columnInfo.CurrentOffset);
        const ui8* data = chunk.Chunk->Quants()->data();

        CB_ENSURE(chunk.Chunk->Quants()->size() - columnInfo.CurrentOffset >= sizeof(ui32));
        const ui32 tokenSize = LittleToHost(ReadUnaligned<ui32>(data + columnInfo.CurrentOffset));
        columnInfo.CurrentOffset += sizeof(ui32);

        CB_ENSURE(chunk.Chunk->Quants()->size() - columnInfo.CurrentOffset >= tokenSize);
        const TString token(reinterpret_cast<const char*>(data + columnInfo.CurrentOffset), tokenSize);
        columnInfo.CurrentOffset += tokenSize;
        ++columnInfo.CurrentDocId;

        if (chunk.Chunk->Quants()->size() == columnInfo.CurrentOffset) {
            columnInfo.CurrentOffset = 0;
            ++columnInfo.CurrentChunkIndex;
        }
        return token;
    }

    const TString TQuantizedPoolColumnsPrinter::GetFloatColumnToken(ui64 docId, EColumn columnType) {
        auto& columnInfo = ColumnsInfo[columnType];

        CB_ENSURE(columnInfo.CurrentDocId == docId, "Only serial lines possible to output.");
        const auto& chunks = QuantizedPool.Chunks[columnInfo.LocalColumnIndex];
        const auto& chunk = chunks[columnInfo.CurrentChunkIndex];

        CB_ENSURE(chunk.Chunk->Quants()->size() > columnInfo.CurrentOffset);
        const ui8* data = chunk.Chunk->Quants()->data();

        CB_ENSURE(chunk.Chunk->Quants()->size() - columnInfo.CurrentOffset >= sizeof(float));
        const TString token(ToString(ReadUnaligned<float>(data + columnInfo.CurrentOffset)));
        columnInfo.CurrentOffset += sizeof(float);
        ++columnInfo.CurrentDocId;

        if (chunk.Chunk->Quants()->size() == columnInfo.CurrentOffset) {
            columnInfo.CurrentOffset = 0;
            ++columnInfo.CurrentChunkIndex;
        }
        return token;
    }

} // namespace NCB
