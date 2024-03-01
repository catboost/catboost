#include "pool_printer.h"

#include "eval_helpers.h"

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/data_util/exists_checker.h>
#include <catboost/private/libs/quantized_pool/serialization.h>

#include <util/generic/xrange.h>
#include <util/string/cast.h>
#include <util/string/split.h>
#include <util/system/byteorder.h>
#include <util/system/unaligned_mem.h>


namespace NCB {
    TDSVPoolColumnsPrinter::TDSVPoolColumnsPrinter(
        TLineDataPoolColumnsPrinterPushArgs&& args
    )
        : LineDataReader(std::move(args.Reader))
        , Delimiter(args.Format.Delimiter)
        , DocId(-1)
        , ColumnsMetaInfo(args.ColumnsMetaInfo)
    {
        UpdateColumnTypeInfo(args.ColumnsMetaInfo);
    }

    TDSVPoolColumnsPrinter::TDSVPoolColumnsPrinter(TPoolColumnsPrinterPullArgs&& args)
        : TDSVPoolColumnsPrinter(
            TLineDataPoolColumnsPrinterPushArgs{
                GetLineDataReader(args.PoolPath, args.Format),
                args.Format,
                args.ColumnsMetaInfo})
    {}


    void TDSVPoolColumnsPrinter::OutputColumnByType(IOutputStream* outStream, ui64 docId, EColumn columnType) {
        CB_ENSURE(
            FromColumnTypeToColumnId.contains(columnType),
            "You can not output " << ToString(columnType) << " column by type");
        *outStream << GetCell(docId, FromColumnTypeToColumnId[columnType]);
    }

    void TDSVPoolColumnsPrinter::OutputFeatureColumnByIndex(IOutputStream* outStream, ui64 docId, ui32 featureId) {
        *outStream << GetCell(docId, FromExternalIdToColumnId[featureId]);
    }

    void TDSVPoolColumnsPrinter::OutputAuxiliaryColumn(
        IOutputStream* outStream,
        ui64 docId,
        ui32 auxiliaryColumnId,
        const TString& /*columnName*/) {
        *outStream << GetCell(docId, auxiliaryColumnId);
    }

    bool TDSVPoolColumnsPrinter::ValidAuxiliaryColumn(const TString& columnName) {
        return AuxiliaryColumnNameToId.contains(columnName);
    }

    ui32 TDSVPoolColumnsPrinter::GetAuxiliaryColumnId(const TString& columnName) {
        return AuxiliaryColumnNameToId[columnName];
    }

    void TDSVPoolColumnsPrinter::UpdateColumnTypeInfo(const TMaybe<TDataColumnsMetaInfo>& columnsMetaInfo) {
        if (columnsMetaInfo.Defined()) {
            for (ui32 columnId : xrange(columnsMetaInfo->Columns.size())) {
                const auto columnType = columnsMetaInfo->Columns[columnId].Type;
                FromColumnTypeToColumnId[columnType] = columnId;
                if (columnType == EColumn::SampleId) {
                    HasDocIdColumn = true;
                }
                if (columnType == EColumn::Auxiliary and columnsMetaInfo->Columns[columnId].Id) {
                    AuxiliaryColumnNameToId[columnsMetaInfo->Columns[columnId].Id] = columnId;
                }
                if (IsFactorColumn(columnType)) {
                    FromExternalIdToColumnId.push_back(columnId);
                }
            }
        }
    }

    std::type_index TDSVPoolColumnsPrinter::GetOutputFeatureType(ui32 featureId) {
        const auto columnType = ColumnsMetaInfo->Columns[FromExternalIdToColumnId[featureId]].Type;
        return columnType == EColumn::Num ? typeid(double) : typeid(TString);
    }

    const TString& TDSVPoolColumnsPrinter::GetCell(ui64 docId, ui32 colId) {
        if (docId == DocId + 1) {
            ++DocId;
            TString line;
            CB_ENSURE(LineDataReader->ReadLine(&line), "there's no line in pool for " << DocId);
            Columns.clear();
            for (const auto& typeName : StringSplitter(line).Split(Delimiter)) {
                Columns.push_back(FromString<TString>(typeName.Token()));
            }
        }
        CB_ENSURE(docId == DocId, "only serial lines possible to output");
        return Columns[colId];
    }

    TQuantizedPoolColumnsPrinter::TQuantizedPoolColumnsPrinter(TPoolColumnsPrinterPullArgs&& args)
        : QuantizedPool(
            LoadQuantizedPool(
                args.PoolPath,
                {/*LockMemory=*/false, /*Precharge=*/false, TDatasetSubset::MakeColumns(!IsSharedFs(args.PoolPath))}))
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
            CB_ENSURE(localColumnIndex < QuantizedPool.Chunks.size(), "Bad localColumnIndex.");
            const auto& chunks = QuantizedPool.Chunks[localColumnIndex];
            auto& chunkIndices = ColumnsInfo[columnType].CorrectChunkOrder;
            chunkIndices.resize(chunks.size());
            Iota(chunkIndices.begin(), chunkIndices.end(), 0);
            Sort(
                chunkIndices,
                [&](ui32 lhs, ui32 rhs) {
                    return chunks[lhs].DocumentOffset < chunks[rhs].DocumentOffset;
                });
            ColumnsInfo[columnType].LocalColumnIndex = localColumnIndex;
        }
    }

    void TQuantizedPoolColumnsPrinter::OutputColumnByType(IOutputStream* outStream, ui64 docId, EColumn columnType) {
        CB_ENSURE(
            ColumnsInfo.contains(columnType),
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
                break;
            default:
                CB_ENSURE("Unsupported output columnType for Quantized pool.");
        }

        *outStream << token;
    }

    void TQuantizedPoolColumnsPrinter::OutputFeatureColumnByIndex(
        IOutputStream* /*outStream*/,
        ui64 /*docId*/,
        ui32 /*columnId*/) {
        CB_ENSURE(false, "Not Implemented for Quantized Pools");
    }

    void TQuantizedPoolColumnsPrinter::OutputAuxiliaryColumn(
        IOutputStream* /*outStream*/,
        ui64 /*docId*/,
        ui32 /*auxiliaryColumnId*/,
        const TString& /*columnName*/) {
        CB_ENSURE(false, "Not Implemented for Quantized Pools");
    }

    bool TQuantizedPoolColumnsPrinter::ValidAuxiliaryColumn(const TString& /*columnName*/) {
        CB_ENSURE(false, "Not Implemented for Quantized Pools");
    }

    ui32 TQuantizedPoolColumnsPrinter::GetAuxiliaryColumnId(const TString& /*columnName*/) {
        CB_ENSURE_INTERNAL(false, "Unreachable");
        return 0;
    }

    std::type_index TQuantizedPoolColumnsPrinter::GetOutputFeatureType(ui32 /*featureId*/) {
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
        const auto& chunk = chunks[columnInfo.CorrectChunkOrder[columnInfo.CurrentChunkIndex]];

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
        const auto& chunk = chunks[columnInfo.CorrectChunkOrder[columnInfo.CurrentChunkIndex]];

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

    TPoolColumnsPrinterLoaderFactory::TRegistrator<TDSVPoolColumnsPrinter> DefPoolColumnsPrinter("");

    TPoolColumnsPrinterLoaderFactory::TRegistrator<TDSVPoolColumnsPrinter> DsvPoolColumnsPrinter("dsv");

    TPoolColumnsPrinterLoaderFactory::TRegistrator<TQuantizedPoolColumnsPrinter> QuantizedPoolColumnsPrinter(
        "quantized");

} // namespace NCB
