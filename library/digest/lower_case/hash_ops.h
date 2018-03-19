#pragma once

#include <util/generic/strbuf.h>

// can be used for caseless hashes like: THashMap<TStringBuf, T, TCIOps, TCIOps>

struct TCIOps {
    size_t operator()(const char* s) const noexcept;
    size_t operator()(const TStringBuf& s) const noexcept;

    bool operator()(const char* f, const char* s) const noexcept;
    bool operator()(const TStringBuf& f, const TStringBuf& s) const noexcept;
};
