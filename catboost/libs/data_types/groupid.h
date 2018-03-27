#pragma once

#include <util/system/types.h>
#include <util/generic/strbuf.h>
#include <util/digest/city.h>

using TGroupId = ui64;

inline TGroupId CalcGroupIdFor(const TStringBuf& token) {
    return CityHash64(token);
}
