#pragma once

#include "path_with_scheme.h"

#include <catboost/private/libs/index_range/index_range.h>

#include <catboost/libs/helpers/exception.h>

#include <library/cpp/object_factory/object_factory.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/ptr.h>

#include <util/stream/file.h>
#include <util/string/escape.h>


namespace NCB {

    struct TDsvFormatOptions {
    public:
        bool HasHeader = false;
        char Delimiter = '\t';
        char NumVectorDelimiter = ';';
        bool IgnoreCsvQuoting = false;

    public:
        explicit TDsvFormatOptions() = default;
        explicit TDsvFormatOptions(
            bool hasHeader,
            char delimiter,
            char numVectorDelimiter = ';',
            bool ignoreCsvQuoting = false
        )
            : HasHeader(hasHeader)
            , Delimiter(delimiter)
            , NumVectorDelimiter(numVectorDelimiter)
            , IgnoreCsvQuoting(ignoreCsvQuoting)
        {
            CB_ENSURE(
                delimiter != numVectorDelimiter,
                "Field delimiter (" << EscapeC(Delimiter) << ") is the same as num vector delimiter ("
                << EscapeC(NumVectorDelimiter) << ')'
            );
        }

        SAVELOAD(HasHeader, Delimiter, NumVectorDelimiter, IgnoreCsvQuoting);
        Y_SAVELOAD_DEFINE(HasHeader, Delimiter, NumVectorDelimiter, IgnoreCsvQuoting);
    };

    struct TLineDataReaderArgs {
        TPathWithScheme PathWithScheme;
        TDsvFormatOptions Format;
        bool KeepLineOrder = true;
    };


    struct ILineDataReader {
        /* if estimate == false, returns number of data lines (w/o header, if present)
           in some cases could be expensive, e.g. for files
           if estimate == true, returns some quick estimation, lower or upper
        */
        virtual ui64 GetDataLineCount(bool estimate = false) = 0;

        /* call before any calls to ReadLine if you need it
           it is an error to call GetHeader after any ReadLine calls
        */
        virtual TMaybe<TString> GetHeader() = 0;

        /* returns true, if data were read
           implementation may return lines in any order
           logical line index is stored to *lineIdx, if lineIdx != nullptr
           not thread-safe
        */
        virtual bool ReadLine(TString* line, ui64* lineIdx = nullptr) = 0;

        virtual ~ILineDataReader() = default;
    };

    using TLineDataReaderFactory =
        NObjectFactory::TParametrizedObjectFactory<ILineDataReader, TString, TLineDataReaderArgs>;

    THolder<ILineDataReader> GetLineDataReader(const TPathWithScheme& pathWithScheme,
                                               const TDsvFormatOptions& format = TDsvFormatOptions(),
                                               bool keepLineOrder = true);


    int CountLines(const TString& poolFile);

    class TFileLineDataReader : public ILineDataReader {
    public:
        explicit TFileLineDataReader(const TLineDataReaderArgs& args)
            : Args(args)
            , IFStream(args.PathWithScheme.Path)
            , HeaderProcessed(!Args.Format.HasHeader)
        {}

        ui64 GetDataLineCount(bool estimate) override {
            if (estimate) {
                // TODO(espetrov): estimate based one first N lines
                return 1;
            }
            ui64 nLines = (ui64)CountLines(Args.PathWithScheme.Path);
            if (Args.Format.HasHeader) {
                --nLines;
            }
            return nLines;
        }

        TMaybe<TString> GetHeader() override {
            if (Args.Format.HasHeader) {
                CB_ENSURE(!HeaderProcessed, "TFileLineDataReader: multiple calls to GetHeader");
                TString header;
                CB_ENSURE(IFStream.ReadLine(header), "TFileLineDataReader: no header in file");
                HeaderProcessed = true;
                return header;
            }

            return {};
        }

        bool ReadLine(TString* line, ui64* lineIdx) override {
            // skip header if it hasn't been read
            if (!HeaderProcessed) {
                GetHeader();
            }
            if (lineIdx) {
                *lineIdx = LineIndex;
            }
            ++LineIndex;
            return IFStream.ReadLine(*line) != 0;
        }

    private:
        TLineDataReaderArgs Args;
        TIFStream IFStream;
        bool HeaderProcessed;
        ui64 LineIndex = 0;
    };

    class TBlocksSubsetLineDataReader final : public ILineDataReader {
    public:
        TBlocksSubsetLineDataReader(THolder<ILineDataReader>&& lineDataReader, TVector<TIndexRange<ui64>>&& subsetBlocks);

        ui64 GetDataLineCount(bool estimate = false) override;

        TMaybe<TString> GetHeader() override;

        bool ReadLine(TString* line, ui64* lineIdx = nullptr) override;

    private:
        THolder<ILineDataReader> LineDataReader;
        TVector<TIndexRange<ui64>> SubsetBlocks;
        ui64 CurrentSubsetBlock;
        ui64 EnclosingLineIdx;
        ui64 LineIdx;
        TString LineBuffer;
    };

    class TIndexedSubsetLineDataReader final : public ILineDataReader {
    public:
        // subsetIndices must be in an increasing order, duplicates are allowed
        // it is not checked in this method for speed
        TIndexedSubsetLineDataReader(THolder<ILineDataReader>&& lineDataReader, TVector<ui64>&& subsetIndices);

        ui64 GetDataLineCount(bool estimate = false) override;

        TMaybe<TString> GetHeader() override;

        bool ReadLine(TString* line, ui64* lineIdx = nullptr) override;

    private:
        THolder<ILineDataReader> LineDataReader;
        TVector<ui64> SubsetIndices;
        TVector<ui64>::iterator CurrentIndex;
        ui64 EnclosingLineIdx;

        TMaybe<TString> Header;
        TString LineBuffer;
    };
}
