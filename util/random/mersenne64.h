#pragma once

#include <util/system/defaults.h>

class IInputStream;

namespace NPrivate {
    class TMersenne64 {
        static constexpr int NN = 312;

    public:
        inline TMersenne64(ui64 s = ULL(19650218))
            : mti(NN + 1)
        {
            InitGenRand(s);
        }

        inline TMersenne64(const ui64* keys, size_t len) noexcept
            : mti(NN + 1)
        {
            InitByArray(keys, len);
        }

        TMersenne64(IInputStream& input);

        inline ui64 GenRand() noexcept {
            if (mti >= NN) {
                InitNext();
            }

            ui64 x = mt[mti++];

            x ^= (x >> 29) & ULL(0x5555555555555555);
            x ^= (x << 17) & ULL(0x71D67FFFEDA60000);
            x ^= (x << 37) & ULL(0xFFF7EEE000000000);
            x ^= (x >> 43);

            return x;
        }

    private:
        void InitNext() noexcept;
        void InitGenRand(ui64 seed) noexcept;
        void InitByArray(const ui64* init_key, size_t key_length) noexcept;

    private:
        ui64 mt[NN];
        int mti;
    };
} // namespace NPrivate
