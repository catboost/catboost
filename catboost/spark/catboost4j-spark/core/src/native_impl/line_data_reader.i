%{
#include <catboost/private/libs/data_util/line_data_reader.h>
%}

%include "typemaps.i"

%include "defaults.i"
%include "maybe.i"
%include "string.i"

%feature("director") NCB::ILineDataReader;

namespace NCB {
    struct TDsvFormatOptions {
    public:
        bool HasHeader;
        char Delimiter;
        char NumVectorDelimiter;
        bool IgnoreCsvQuoting;
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
}


