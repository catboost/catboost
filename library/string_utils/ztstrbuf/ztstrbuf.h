#pragma once

#include <util/generic/strbuf.h>

/*
 * Zero-terminated string view.
 *
 * Has a c_str() for use with system/cstdlib calls (like TString)
 * but can be constructed from a string literal or command-line arg
 * without memory allocation (like TStringBuf).
 *
 * Use it to reference filenames, thread names, string formats etc.
 */

class TZtStringBuf: public TStringBuf {
public:
    TZtStringBuf(const char* s)
        : TStringBuf(s)
    {
    }

    TZtStringBuf(const TString& s)
        : TStringBuf(s)
    {
    }

    TZtStringBuf()
        : TZtStringBuf(TString{})
    {
    }

    const char* c_str() const {
        return data();
    }
};
