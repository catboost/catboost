#include "parsed_request.h"

#include <util/string/strip.h>
#include <util/generic/yexception.h>
#include <util/string/cast.h>

static inline TStringBuf StripLeft(const TStringBuf& s) noexcept {
    const char* b = s.begin();
    const char* e = s.end();

    StripRangeBegin(b, e);

    return TStringBuf(b, e);
}

TParsedHttpRequest::TParsedHttpRequest(const TStringBuf& str) {
    TStringBuf tmp;

    if (!StripLeft(str).TrySplit(' ', Method, tmp)) {
        ythrow yexception() << "bad request(" << ToString(str).Quote() << ")";
    }

    if (!StripLeft(tmp).TrySplit(' ', Request, Proto)) {
        ythrow yexception() << "bad request(" << ToString(str).Quote() << ")";
    }

    Proto = StripLeft(Proto);
}

TParsedHttpLocation::TParsedHttpLocation(const TStringBuf& req) {
    req.Split('?', Path, Cgi);
}
