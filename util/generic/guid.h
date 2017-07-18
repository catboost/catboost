#pragma once

#include "fwd.h"

#include <util/str_stl.h>

/**
 * GUID (UUID) generation
 * https://en.wikipedia.org/wiki/Universally_unique_identifier
 * https://en.wikipedia.org/wiki/Globally_unique_identifier
 **/

struct TGUID {
    ui32 dw[4] = {};

    constexpr TGUID() {
    }

    constexpr bool IsEmpty() const noexcept {
        return (dw[0] | dw[1] | dw[2] | dw[3]) == 0;
    }

    constexpr explicit operator bool() const noexcept {
        return !IsEmpty();
    }
};

constexpr bool operator==(const TGUID& a, const TGUID& b) noexcept {
    return a.dw[0] == b.dw[0] && a.dw[1] == b.dw[1] && a.dw[2] == b.dw[2] && a.dw[3] == b.dw[3];
}

constexpr bool operator!=(const TGUID& a, const TGUID& b) noexcept {
    return !(a == b);
}

struct TGUIDHash {
    constexpr int operator()(const TGUID& a) const noexcept {
        return a.dw[0] + a.dw[1] + a.dw[2] + a.dw[3];
    }
};

template <>
struct THash<TGUID> {
    constexpr size_t operator()(const TGUID& g) const noexcept {
        return (unsigned int)TGUIDHash()(g);
    }
};

void CreateGuid(TGUID* res);
TString GetGuidAsString(const TGUID& g);
TString CreateGuidAsString();
TGUID GetGuid(const TString& s);
bool GetGuid(const TString& s, TGUID& result);
