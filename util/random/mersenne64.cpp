#include "mersenne64.h"

#include <util/generic/array_size.h>
#include <util/stream/input.h>

#define MM 156
#define MATRIX_A ULL(0xB5026F5AA96619E9)
#define UM ULL(0xFFFFFFFF80000000)
#define LM ULL(0x7FFFFFFF)

using namespace NPrivate;

void TMersenne64::InitGenRand(ui64 seed) noexcept {
    mt[0] = seed;

    for (mti = 1; mti < NN; ++mti) {
        mt[mti] = (ULL(6364136223846793005) * (mt[mti - 1] ^ (mt[mti - 1] >> 62)) + mti);
    }
}

void TMersenne64::InitByArray(const ui64* init_key, size_t key_length) noexcept {
    ui64 i = 1;
    ui64 j = 0;
    ui64 k;

    InitGenRand(ULL(19650218));

    k = NN > key_length ? NN : key_length;

    for (; k; --k) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 62)) * ULL(3935559000370003845))) + init_key[j] + j;

        ++i;
        ++j;

        if (i >= NN) {
            mt[0] = mt[NN - 1];
            i = 1;
        }

        if (j >= key_length) {
            j = 0;
        }
    }

    for (k = NN - 1; k; --k) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 62)) * ULL(2862933555777941757))) - i;

        ++i;

        if (i >= NN) {
            mt[0] = mt[NN - 1];
            i = 1;
        }
    }

    mt[0] = ULL(1) << 63;
}

void TMersenne64::InitNext() noexcept {
    int i;
    ui64 x;
    ui64 mag01[2] = {
        ULL(0),
        MATRIX_A,
    };

    if (mti == NN + 1) {
        InitGenRand(ULL(5489));
    }

    for (i = 0; i < NN - MM; ++i) {
        x = (mt[i] & UM) | (mt[i + 1] & LM);
        mt[i] = mt[i + MM] ^ (x >> 1) ^ mag01[(int)(x & ULL(1))];
    }

    for (; i < NN - 1; ++i) {
        x = (mt[i] & UM) | (mt[i + 1] & LM);
        mt[i] = mt[i + (MM - NN)] ^ (x >> 1) ^ mag01[(int)(x & ULL(1))];
    }

    x = (mt[NN - 1] & UM) | (mt[0] & LM);
    mt[NN - 1] = mt[MM - 1] ^ (x >> 1) ^ mag01[(int)(x & ULL(1))];

    mti = 0;
}

TMersenne64::TMersenne64(IInputStream& input)
    : mti(NN + 1)
{
    ui64 buf[128];

    input.LoadOrFail(buf, sizeof(buf));
    InitByArray(buf, Y_ARRAY_SIZE(buf));
}

#undef MM
#undef MATRIX_A
#undef UM
#undef LM
