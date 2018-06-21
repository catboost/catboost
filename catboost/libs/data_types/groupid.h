#pragma once

#include <util/system/types.h>
#include <util/generic/strbuf.h>
#include <util/digest/city.h>

using TGroupId = ui64;

inline TGroupId CalcGroupIdFor(const TStringBuf& token) {
    return CityHash64(token);
}

using TSubgroupId = ui32;

inline TSubgroupId CalcSubgroupIdFor(const TStringBuf& token) {
    return static_cast<TSubgroupId>(CityHash64(token));
}
