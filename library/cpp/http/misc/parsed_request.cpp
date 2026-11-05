#include "parsed_request.h"

#include <util/string/strip.h>
#include <util/generic/yexception.h>
#include <util/string/cast.h>

static TStringBuf StripLeft(const TStringBuf s Y_LIFETIME_BOUND) noexcept {
    auto b = s.begin();
    auto e = s.end();

    StripRangeBegin(b, e);

    return TStringBuf(b, e);
}

TParsedHttpRequest::TParsedHttpRequest(const TStringBuf str Y_LIFETIME_BOUND) {
    TStringBuf tmp;

    if (!StripLeft(str).TrySplit(' ', Method, tmp)) {
        ythrow yexception() << "bad request(" << ToString(str).Quote() << ")";
    }

    if (!StripLeft(tmp).TrySplit(' ', Request, Proto)) {
        ythrow yexception() << "bad request(" << ToString(str).Quote() << ")";
    }

    Proto = StripLeft(Proto);
}

TParsedHttpLocation::TParsedHttpLocation(const TStringBuf req Y_LIFETIME_BOUND) {
    req.Split('?', Path, Cgi);
}
