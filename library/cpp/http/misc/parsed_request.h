#pragma once

#include <util/generic/strbuf.h>

struct TParsedHttpRequest {
    TParsedHttpRequest(const TStringBuf str Y_LIFETIME_BOUND);

    TStringBuf Method;
    TStringBuf Request;
    TStringBuf Proto;
};

struct TParsedHttpLocation {
    TParsedHttpLocation(const TStringBuf req Y_LIFETIME_BOUND);

    TStringBuf Path;
    TStringBuf Cgi;
};

struct TParsedHttpFull: public TParsedHttpRequest, public TParsedHttpLocation {
    TParsedHttpFull(const TStringBuf line Y_LIFETIME_BOUND)
        : TParsedHttpRequest(line)
        , TParsedHttpLocation(Request)
    {
    }
};
