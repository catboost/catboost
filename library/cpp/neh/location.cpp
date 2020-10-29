#include "location.h"

#include <util/string/cast.h>

using namespace NNeh;

TParsedLocation::TParsedLocation(TStringBuf path) {
    path.Split(':', Scheme, path);
    path.Skip(2);

    const size_t pos = path.find_first_of(TStringBuf("?@"));

    if (TStringBuf::npos != pos && '@' == path[pos]) {
        path.SplitAt(pos, UserInfo, path);
        path.Skip(1);
    }

    auto checkRange = [](size_t b, size_t e){
        return b != TStringBuf::npos && e != TStringBuf::npos && b < e;
    };

    size_t oBracket = path.find_first_of('[');
    size_t cBracket = path.find_first_of(']');
    size_t endEndPointPos = path.find_first_of('/');
    if (checkRange(oBracket, cBracket)) {
        endEndPointPos = path.find_first_of('/', cBracket);
    }
    EndPoint = path.SubStr(0, endEndPointPos);
    Host = EndPoint;

    size_t lastColon = EndPoint.find_last_of(':');
    if (checkRange(cBracket, lastColon)
        || (cBracket == TStringBuf::npos && lastColon != TStringBuf::npos))
    {
        Host = EndPoint.SubStr(0, lastColon);
        Port = EndPoint.SubStr(lastColon + 1, EndPoint.size() - lastColon + 1);
    }

    if (endEndPointPos != TStringBuf::npos) {
        Service = path.SubStr(endEndPointPos + 1, path.size() - endEndPointPos + 1);
    }
}

ui16 TParsedLocation::GetPort() const {
    if (!Port) {
        return TStringBuf("https") == Scheme || TStringBuf("fulls") == Scheme || TStringBuf("posts") == Scheme ? 443 : 80;
    }

    return FromString<ui16>(Port);
}
