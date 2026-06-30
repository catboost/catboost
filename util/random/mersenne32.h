#pragma once

#include <util/system/defaults.h>

class IInputStream;

namespace NPrivate {
    class TMersenne32 {
        static constexpr int N = 624;

    public:
        inline TMersenne32(ui32 s = 19650218UL) noexcept
            : mti(N + 1)
        {
            InitGenRand(s);
        }

        inline TMersenne32(const ui32* init_key, size_t key_length) noexcept
            : mti(N + 1)
        {
            InitByArray(init_key, key_length);
        }

        TMersenne32(IInputStream& input);

        inline ui32 GenRand() noexcept {
            if (mti >= N) {
                InitNext();
            }

            ui32 y = mt[mti++];

            y ^= (y >> 11);
            y ^= (y << 7) & 0x9d2c5680UL;
            y ^= (y << 15) & 0xefc60000UL;
            y ^= (y >> 18);

            return y;
        }

    private:
        void InitNext() noexcept;
        void InitGenRand(ui32 s) noexcept;
        void InitByArray(const ui32* init_key, size_t key_length) noexcept;

    private:
        ui32 mt[N];
        int mti;
    };
} // namespace NPrivate
