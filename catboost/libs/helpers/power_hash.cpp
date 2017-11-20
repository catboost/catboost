#include "power_hash.h"

#include <util/generic/vector.h>
#include <util/generic/ymath.h>

namespace NPowerHash2 {
    const int FAST_POW_TABLE_SIZE = 10;

    TVector<TFastPowTable> FastPowTable;

    static bool IsPrime(ui64 n) {
        if (n != 2 && (n & 1) == 0) {
            return false;
        }
        ui64 maxDiv = Min((ui64)1000000, (ui64)sqrt(n * 1.));
        for (ui64 dd = 3; dd <= maxDiv; ++dd) {
            if ((n % dd) == 0) {
                return false;
            }
        }
        return true;
    }

    static struct TInitPowerHash {
        TInitPowerHash() {
            FastPowTable.resize(FAST_POW_TABLE_SIZE);
            ui64 myPrime = 653;
            for (int i = 0; i < FastPowTable.ysize(); ++i) {
                for (int tableId = 0; tableId < TFastPowTable::TABLE_COUNT; ++tableId) {
                    for (;;) {
                        myPrime += 2;
                        if (IsPrime(myPrime)) {
                            break;
                        }
                    }
                    FastPowTable[i].Init(tableId, myPrime);
                }
            }
        }
    } InitPowerHash;
}
