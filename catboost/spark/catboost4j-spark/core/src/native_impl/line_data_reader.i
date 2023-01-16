%{
#include <catboost/private/libs/data_util/line_data_reader.h>
%}

%include "typemaps.i"

%include "defaults.i"
%include "maybe.i"
%include "string.i"

%feature("director") NCB::ILineDataReader;

%catches(yexception) ILineDataReader::GetDataLineCount(bool estimate = false);
%catches(yexception) ILineDataReader::GetHeader();
%catches(yexception) ILineDataReader::ReadLine(TString* line, ui64* lineIdx = nullptr);
%catches(yexception) ILineDataReader::ReadLine(TString*, TString*, ui64* lineIdx = nullptr);

namespace NCB {
    struct TDsvFormatOptions {
    public:
        bool HasHeader;
        char Delimiter;
        char NumVectorDelimiter;
        bool IgnoreCsvQuoting;
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
        virtual bool ReadLine(TString*, TString*, ui64* lineIdx = nullptr) = 0;
        virtual ~ILineDataReader() = default;
    };
}


