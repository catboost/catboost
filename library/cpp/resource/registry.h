#pragma once

#include <util/generic/string.h>
#include <util/generic/strbuf.h>

#include "resource.h"

namespace NResource {
    TString Compress(const TStringBuf data);
    TString Decompress(const TStringBuf data);

    class IMatch {
    public:
        virtual void OnMatch(const TResource& res) = 0;
        virtual ~IMatch() = default;
    };

    class IStore {
    public:
        virtual void Store(const TStringBuf key, const TStringBuf data) = 0;
        virtual bool Has(const TStringBuf key) const = 0;
        virtual bool FindExact(const TStringBuf key, TString* out) const = 0;
        virtual void FindMatch(const TStringBuf subkey, IMatch& cb) const = 0;
        virtual size_t Count() const noexcept = 0;
        virtual TStringBuf KeyByIndex(size_t idx) const = 0;
        virtual ~IStore() = default;
    };

    IStore* CommonStore();

    struct TRegHelper {
        TRegHelper(const TStringBuf key, const TStringBuf data);
    };
}
