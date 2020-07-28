#pragma once

#include <util/generic/strbuf.h>

namespace NNeh {
    struct TParsedLocation {
        TParsedLocation(TStringBuf path);

        ui16 GetPort() const;

        TStringBuf Scheme;
        TStringBuf UserInfo;
        TStringBuf EndPoint;
        TStringBuf Host;
        TStringBuf Port;
        TStringBuf Service;
    };
}
