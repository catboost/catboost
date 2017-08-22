#include "mersenne32.h"

#include <util/generic/array_size.h>
#include <util/stream/input.h>

using namespace NPrivate;

#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

void TMersenne32::InitGenRand(ui32 s) noexcept {
    mt[0] = s;

    for (mti = 1; mti < N; ++mti) {
        mt[mti] = (1812433253UL * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
    }
}

void TMersenne32::InitByArray(const ui32 init_key[], size_t key_length) noexcept {
    InitGenRand(19650218UL);

    ui32 i = 1;
    ui32 j = 0;
    ui32 k = ui32(N > key_length ? N : key_length);

    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525UL)) + init_key[j] + j;

        ++i;
        ++j;

        if (i >= N) {
            mt[0] = mt[N - 1];
            i = 1;
        }

        if (j >= key_length) {
            j = 0;
        }
    }

    for (k = N - 1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941UL)) - i;

        ++i;

        if (i >= N) {
            mt[0] = mt[N - 1];
            i = 1;
        }
    }

    mt[0] = 0x80000000UL;
}

void TMersenne32::InitNext() noexcept {
    int kk;
    ui32 y;
    ui32 mag01[2] = {
        0x0UL,
        MATRIX_A,
    };

    if (mti == N + 1) {
        InitGenRand(5489UL);
    }

    for (kk = 0; kk < N - M; ++kk) {
        y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
        mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }

    for (; kk < N - 1; ++kk) {
        y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
        mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }

    y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
    mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];

    mti = 0;
}

TMersenne32::TMersenne32(IInputStream& input)
    : mti(N + 1)
{
    ui32 buf[128];

    input.LoadOrFail(buf, sizeof(buf));
    InitByArray(buf, Y_ARRAY_SIZE(buf));
}

#undef M
#undef MATRIX_A
#undef UPPER_MASK
#undef LOWER_MASK
