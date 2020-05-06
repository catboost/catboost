#pragma once

#include <util/generic/strbuf.h>

template <bool addAll, char sep, char sepKeyVal, class F>
static inline void ScanKeyValue(TStringBuf s, F&& f) {
    TStringBuf key, val;

    while (!s.empty()) {
        val = s.NextTok(sep);

        if (val.empty()) {
            continue; // && case
        }

        key = val.NextTok(sepKeyVal);

        if (addAll || val.IsInited()) {
            f(key, val); // includes empty keys
        }
    }
}
