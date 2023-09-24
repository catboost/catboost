#include "line_data_reader.h"

#include <util/system/compiler.h>
#include <util/system/fs.h>


namespace NCB {

    THolder<ILineDataReader> GetLineDataReader(const TPathWithScheme& pathWithScheme,
                                               const TDsvFormatOptions& format,
                                               bool keepLineOrder)
    {
        return GetProcessor<ILineDataReader, TLineDataReaderArgs>(
            pathWithScheme, TLineDataReaderArgs{pathWithScheme, format, keepLineOrder}
        );
    }

    int CountLines(const TString& poolFile) {
        CB_ENSURE(NFs::Exists(TString(poolFile)), "pool file '" << TString(poolFile) << "' is not found");
        TIFStream reader(poolFile.c_str());
        size_t count = 0;
        TString buffer;
        while (reader.ReadLine(buffer)) {
            ++count;
        }
        return count;
    }


    TBlocksSubsetLineDataReader::TBlocksSubsetLineDataReader(THolder<ILineDataReader>&& lineDataReader, TVector<TIndexRange<ui64>>&& subsetBlocks)
        : LineDataReader(std::move(lineDataReader))
        , SubsetBlocks(std::move(subsetBlocks))
        , CurrentSubsetBlock(0)
        , EnclosingLineIdx(0)
        , LineIdx(0)
    {}

    ui64 TBlocksSubsetLineDataReader::GetDataLineCount(bool estimate) {
        Y_UNUSED(estimate);

        ui64 result = 0;
        for (const auto& block : SubsetBlocks) {
            result += block.GetSize();
        }
        return result;
    }

    TMaybe<TString> TBlocksSubsetLineDataReader::GetHeader() {
        return LineDataReader->GetHeader();
    }

    bool TBlocksSubsetLineDataReader::ReadLine(TString* line, ui64* lineIdx) {
        while (true) {
            if (CurrentSubsetBlock >= SubsetBlocks.size()) {
                return false;
            }
            if (EnclosingLineIdx < SubsetBlocks[CurrentSubsetBlock].End) {
                bool enclosingReadResult = LineDataReader->ReadLine(&LineBuffer);
                Y_ASSERT(enclosingReadResult);

                if (EnclosingLineIdx >= SubsetBlocks[CurrentSubsetBlock].Begin) {
                    // in a subset
                    *line = std::move(LineBuffer);
                    if (lineIdx) {
                        *lineIdx = LineIdx;
                    }
                    ++LineIdx;
                    ++EnclosingLineIdx;
                    return true;
                } else {
                    ++EnclosingLineIdx;
                }
            } else {
                ++CurrentSubsetBlock;
            }
        }
    }


    TIndexedSubsetLineDataReader::TIndexedSubsetLineDataReader(THolder<ILineDataReader>&& lineDataReader, TVector<ui64>&& subsetIndices)
        : LineDataReader(std::move(lineDataReader))
        , SubsetIndices(std::move(subsetIndices))
        , CurrentIndex(SubsetIndices.begin())
        , EnclosingLineIdx(0)
        , Header(LineDataReader->GetHeader())
    {
        if (!SubsetIndices.empty()) {
            bool enclosingReadResult = LineDataReader->ReadLine(&LineBuffer);
            CB_ENSURE(enclosingReadResult, "Reached the end of data but not reached the end of subset");
        }
    }

    ui64 TIndexedSubsetLineDataReader::GetDataLineCount(bool estimate) {
        Y_UNUSED(estimate);
        return SubsetIndices.size();
    }

    TMaybe<TString> TIndexedSubsetLineDataReader::GetHeader() {
        return Header;
    }

    bool TIndexedSubsetLineDataReader::ReadLine(TString* line, ui64* lineIdx) {
        if (CurrentIndex == SubsetIndices.end()) {
            return false;
        }

        while (true) {
            if (EnclosingLineIdx == *CurrentIndex) {
                // in a subset

                if (lineIdx) {
                    *lineIdx = (CurrentIndex - SubsetIndices.begin());
                }

                auto prevIndex = *CurrentIndex;
                ++CurrentIndex;
                if ((CurrentIndex == SubsetIndices.end()) || (*CurrentIndex != prevIndex)) {
                    *line = std::move(LineBuffer);
                } else {
                    *line = LineBuffer;
                }
                return true;
            }

            bool enclosingReadResult = LineDataReader->ReadLine(&LineBuffer);
            CB_ENSURE(enclosingReadResult, "Reached the end of data but not reached the end of subset");
            ++EnclosingLineIdx;
        }
    }


    TLineDataReaderFactory::TRegistrator<TFileLineDataReader> DefLineDataReaderReg("");
    TLineDataReaderFactory::TRegistrator<TFileLineDataReader> FileLineDataReaderReg("file");
    TLineDataReaderFactory::TRegistrator<TFileLineDataReader> DsvLineDataReaderReg("dsv");
}
