#pragma once

#include <util/generic/bitops.h>
#include <util/stream/str.h>
#include <util/system/align.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace NBitUtils {
    // todo: use builtins where possible

    Y_FORCE_INLINE ui64 BytesUp(ui64 bits) {
        return (bits + 7ULL) >> 3ULL;
    }

    extern const ui64 FLAG_MASK[];

    Y_FORCE_INLINE ui64 Flag(ui64 bit) {
        return FLAG_MASK[bit];
    }

    // FlagK<0> => 1, FlagK<1> => 10, ... FlagK<64> => 0
    template <ui64 bits>
    Y_FORCE_INLINE ui64 FlagK() {
        return bits >= 64 ? 0 : 1ULL << (bits & 63);
    }

    // MaskK<0> => 0, MaskK<1> => 1, ... MaskK<64> => -1
    template <ui64 bits>
    Y_FORCE_INLINE ui64 MaskK() {
        return FlagK<bits>() - 1;
    }

    template <ui64 bits>
    Y_FORCE_INLINE ui64 MaskK(ui64 skipbits) {
        return MaskK<bits>() << skipbits;
    }
}
