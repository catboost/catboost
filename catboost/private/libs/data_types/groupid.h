#pragma once

#include <util/system/types.h>
#include <util/generic/bitops.h>
#include <util/generic/strbuf.h>
#include <util/digest/city.h>
#include <util/string/cast.h>

using TGroupId = ui64;

inline TGroupId CalcGroupIdFor(const TStringBuf& token) noexcept {
    TGroupId groupId;
    if (!token.empty() && token[0] != '0' && AllOf(token, ::isdigit) && TryFromString<ui64>(token, groupId)) {
        return ReverseBits(groupId);
    }
    return CityHash64(token);
}

using TSubgroupId = ui32;

inline TSubgroupId CalcSubgroupIdFor(const TStringBuf& token) noexcept {
    return static_cast<TSubgroupId>(CityHash64(token));
}
