#pragma once

#include "path_with_scheme.h"

#include <catboost/libs/helpers/exception.h>

#include <library/cpp/object_factory/object_factory.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>

#include <util/stream/file.h>
#include <util/string/escape.h>


namespace NCB {

    struct TDsvFormatOptions {
    public:
        bool HasHeader;
        char Delimiter;
        char NumVectorDelimiter;
        bool IgnoreCsvQuoting;

    public:
        explicit TDsvFormatOptions(
            bool hasHeader = false,
            char delimiter = '\t',
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
    };


    struct ILineDataReader {
        /* returns number of data lines (w/o header, if present)
           in some cases (e.g. for files, could be expensive)
        */
        virtual ui64 GetDataLineCount() = 0;

        /* call before any calls to NextLine if you need it
           it is an error to call GetHeader after any ReadLine calls
        */
        virtual TMaybe<TString> GetHeader() = 0;

        /* returns true if there still was some data
           not thread-safe
        */
        virtual bool ReadLine(TString* line) = 0;

        virtual bool ReadLine(TString*, TString*) = 0;

        virtual ~ILineDataReader() = default;
    };

    using TLineDataReaderFactory =
        NObjectFactory::TParametrizedObjectFactory<ILineDataReader, TString, TLineDataReaderArgs>;

    THolder<ILineDataReader> GetLineDataReader(const TPathWithScheme& pathWithScheme,
                                               const TDsvFormatOptions& format = TDsvFormatOptions());


    int CountLines(const TString& poolFile);

    class TFileLineDataReader : public ILineDataReader {
    public:
        TFileLineDataReader(const TLineDataReaderArgs& args)
            : Args(args)
            , IFStream(args.PathWithScheme.Path)
            , HeaderProcessed(!Args.Format.HasHeader)
        {}

        ui64 GetDataLineCount() override {
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

        bool ReadLine(TString* line) override {
            // skip header if it hasn't been read
            if (!HeaderProcessed) {
                GetHeader();
            }
            return IFStream.ReadLine(*line) != 0;
        }

        bool ReadLine(TString*, TString*) override {
            Y_UNREACHABLE();
        }

    private:
        TLineDataReaderArgs Args;
        TIFStream IFStream;
        bool HeaderProcessed;
    };

}
