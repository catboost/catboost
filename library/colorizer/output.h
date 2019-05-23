#pragma once

#include "colors.h"

// Note: this is an old interface for printing colors to stream.
// Consider printing elements of `EAnsiCode` directly.

namespace NColorizer {
    typedef TStringBuf (TColors::*TColorFunc)() const;

    struct TColorHandle {
        const TColors* C;
        TColorFunc F;

        inline TColorHandle(const TColors* c, TColorFunc f) noexcept
            : C(c)
            , F(f)
        {
        }
    };

#define DEF(X)                                              \
    static inline TColorHandle X() noexcept {               \
        return TColorHandle(&StdErr(), &TColors::X##Color); \
    }

    DEF(Old)
    DEF(Black)
    DEF(Green)
    DEF(Cyan)
    DEF(Red)
    DEF(Purple)
    DEF(Brown)
    DEF(LightGray)
    DEF(DarkGray)
    DEF(LightBlue)
    DEF(LightGreen)
    DEF(LightCyan)
    DEF(LightRed)
    DEF(LightPurple)
    DEF(Yellow)
    DEF(White)

#undef DEF
}
