#include "lcg_engine.h"

namespace NPrivate {
    template <typename T>
    T LcgAdvance(T seed, T lcgBase, T lcgAddend, T delta) noexcept {
        // seed[n+1] = A * seed[n] + B, A = lcgBase, B = lcgAddend
        // seed[n] = A**n * seed[0] + (A**n - 1) / (A - 1) * B
        // (initial value of n) = m * 2**k + (lower bits of n)
        T mask = 1;
        while (mask != (1ULL << (8 * sizeof(T) - 1)) && (mask << 1) <= delta) {
            mask <<= 1;
        }
        T apow = 1; // A**m
        T adiv = 0; // (A**m-1)/(A-1)
        for (; mask; mask >>= 1) {
            // m *= 2
            adiv *= apow + 1;
            apow *= apow;
            if (delta & mask) {
                // m++
                adiv += apow;
                apow *= lcgBase;
            }
        }
        return seed * apow + lcgAddend * adiv;
    }

    template ui32 LcgAdvance<ui32>(ui32, ui32, ui32, ui32) noexcept;
    template ui64 LcgAdvance<ui64>(ui64, ui64, ui64, ui64) noexcept;
}
