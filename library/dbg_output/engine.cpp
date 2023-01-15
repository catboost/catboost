#include "engine.h"

#include <util/string/cast.h>
#include <util/string/escape.h>

#if !defined(DBGDUMP_INLINE_IF_INCLUDED)
#define DBGDUMP_INLINE_IF_INCLUDED
#endif

DBGDUMP_INLINE_IF_INCLUDED void TDumpBase::String(const TStringBuf& s) {
    if (s) {
        Raw(TString(s).Quote());
    } else {
        Raw("(empty)");
    }
}

DBGDUMP_INLINE_IF_INCLUDED void TDumpBase::String(const TWtringBuf& s) {
    Raw("w");
    String(ToString(s));
}

DBGDUMP_INLINE_IF_INCLUDED void TDumpBase::Raw(const TStringBuf& s) {
    Stream().Write(s.data(), s.size());
}

DBGDUMP_INLINE_IF_INCLUDED void TDumpBase::Char(char ch) {
    Raw("'" + EscapeC(&ch, 1) + "'");
}

DBGDUMP_INLINE_IF_INCLUDED void TDumpBase::Char(wchar16 ch) {
    Raw("w'" + ToString(EscapeC(&ch, 1)) + "'");
}
