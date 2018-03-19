#include "location.h"

#include <util/string/cast.h>

using namespace NNeh;

TParsedLocation::TParsedLocation(TStringBuf path) {
    path.Split(':', Scheme, path);
    path.Skip(2);

    const size_t pos = path.find_first_of(STRINGBUF("?@"));

    if (TStringBuf::npos != pos && '@' == path[pos]) {
        path.SplitAt(pos, UserInfo, path);
        path.Skip(1);
    }

    path.Split('/', EndPoint, Service);

    if (!EndPoint.TryRSplit(':', Host, Port)) {
        Host = EndPoint;
    }
}

ui16 TParsedLocation::GetPort() const {
    if (!Port) {
        return STRINGBUF("https") == Scheme || STRINGBUF("fulls") == Scheme || STRINGBUF("posts") == Scheme ? 443 : 80;
    }

    return FromString<ui16>(Port);
}
