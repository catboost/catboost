#pragma once

#include "http.h"

#include <library/cpp/http/misc/httpreqdata.h>

class THttpClientRequestExtension: public TClientRequest {
public:
    bool Parse(char* req, TBaseServerRequestData& rd);
    bool ProcessHeaders(TBaseServerRequestData& rd, TBlob& postData);
protected:
    virtual bool OptionsAllowed() {
        return false;
    }
};

template <class TRequestData>
class THttpClientRequestExtImpl: public THttpClientRequestExtension {
protected:
    bool Parse(char* req) {
        return THttpClientRequestExtension::Parse(req, RD);
    }
    bool ProcessHeaders() {
        return THttpClientRequestExtension::ProcessHeaders(RD, Buf);
    }

protected:
    TRequestData RD;
    TBlob Buf;
};

using THttpClientRequestEx = THttpClientRequestExtImpl<TServerRequestData>;
