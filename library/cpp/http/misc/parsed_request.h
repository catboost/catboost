#pragma once

#include <util/generic/strbuf.h>

struct TParsedHttpRequest {
    TParsedHttpRequest(const TStringBuf& str);

    TStringBuf Method;
    TStringBuf Request;
    TStringBuf Proto;
};

struct TParsedHttpLocation {
    TParsedHttpLocation(const TStringBuf& req);

    TStringBuf Path;
    TStringBuf Cgi;
};

struct TParsedHttpFull: public TParsedHttpRequest, public TParsedHttpLocation {
    inline TParsedHttpFull(const TStringBuf& line)
        : TParsedHttpRequest(line)
        , TParsedHttpLocation(Request)
    {
    }
};
