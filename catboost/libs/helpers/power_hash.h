#pragma once

#include <util/generic/vector.h>

namespace NPowerHash2 {
    struct TFastPowTable {
        enum {
            BITS = 8,
            TABLE_COUNT = 64 / BITS
        };
        ui64 PowTable[TABLE_COUNT][1 << BITS];

        void Init(int tableId, ui64 base) {
            ui64 res = 1;
            for (int i = 0; i < (1 << BITS); ++i) {
                PowTable[tableId][i] = res;
                res *= base;
            }
        }
    };

    union TIntegerPowerUnion {
        ui64 Power8;
        struct {
            ui8 X[8];
        } Power1;
    };

    inline ui64 IntegerPow(const TFastPowTable& tbl, ui64 powArg) {
        static_assert(TFastPowTable::BITS == 8, "expect TFastPowTable::BITS == 8");
        TIntegerPowerUnion pow;
        pow.Power8 = powArg;
        if (powArg & 0xffffffff00000000ull) {
            return
                (tbl.PowTable[0][pow.Power1.X[0]] * tbl.PowTable[1][pow.Power1.X[1]]) *
                (tbl.PowTable[2][pow.Power1.X[2]] * tbl.PowTable[3][pow.Power1.X[3]]) *
                (tbl.PowTable[4][pow.Power1.X[4]] * tbl.PowTable[5][pow.Power1.X[5]]) *
                (tbl.PowTable[6][pow.Power1.X[6]] * tbl.PowTable[7][pow.Power1.X[7]]);

        } else {
            if (powArg & 0xffff0000ull) {
                return
                    (tbl.PowTable[0][pow.Power1.X[0]] * tbl.PowTable[1][pow.Power1.X[1]]) *
                    (tbl.PowTable[2][pow.Power1.X[2]] * tbl.PowTable[3][pow.Power1.X[3]]);

            } else {
                return tbl.PowTable[0][pow.Power1.X[0]] * tbl.PowTable[1][pow.Power1.X[1]];
            }
        }
    }

    extern TVector<TFastPowTable> FastPowTable;

    // if we take low bits of this number we will often get a prime number
    // there are no 111 and 000 patterns in binary representation of this number
    const ui64 MAGIC_MULT = 0x4906ba494954cb65ull;

    inline ui64 CalcHash(ui64 a, ui64 b) {
        return MAGIC_MULT * (a + MAGIC_MULT * b);
        //return IntegerPow(FastPowTable[0], a) * IntegerPow(FastPowTable[1], b);
    }
}
