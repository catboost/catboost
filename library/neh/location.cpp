#include "location.h"

#include <util/string/cast.h>

using namespace NNeh;

TParsedLocation::TParsedLocation(TStringBuf path) {
    path.Split(':', Scheme, path);
    path.Skip(2);

    const size_t pos = path.find_first_of(AsStringBuf("?@"));

    if (TStringBuf::npos != pos && '@' == path[pos]) {
        path.SplitAt(pos, UserInfo, path);
        path.Skip(1);
    }

    if (!path.TrySplit('/', EndPoint, Service)) {
        EndPoint = path;
        Service = "";
    }

    if (!EndPoint.TryRSplit(':', Host, Port)) {
        Host = EndPoint;
    }
}

ui16 TParsedLocation::GetPort() const {
    if (!Port) {
        return AsStringBuf("https") == Scheme || AsStringBuf("fulls") == Scheme || AsStringBuf("posts") == Scheme ? 443 : 80;
    }

    return FromString<ui16>(Port);
}
