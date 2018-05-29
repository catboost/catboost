#pragma once

#include "path_with_scheme.h"

#include <library/object_factory/object_factory.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>



namespace NCB {

    struct TDsvFormatOptions {
        bool HasHeader = false;
        char Delimiter = '\t';
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

        virtual ~ILineDataReader() = default;
    };

    using TLineDataReaderFactory =
        NObjectFactory::TParametrizedObjectFactory<ILineDataReader, TString, TLineDataReaderArgs>;

    THolder<ILineDataReader> GetLineDataReader(const TPathWithScheme& pathWithScheme,
                                               const TDsvFormatOptions& format = {});

}
