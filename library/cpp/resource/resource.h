#pragma once

#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>

namespace NResource {
    struct TResource {
        TStringBuf Key;
        TString Data;
    };

    typedef TVector<TResource> TResources;

    bool Has(const TStringBuf key);
    TString Find(const TStringBuf key);
    bool FindExact(const TStringBuf key, TString* out);
    /// @note Perform full scan for now.
    void FindMatch(const TStringBuf subkey, TResources* out);
    size_t Count() noexcept;
    TStringBuf KeyByIndex(size_t idx);
    TVector<TStringBuf> ListAllKeys();
}
