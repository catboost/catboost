#include "packers.h"
#include "region_packer.h"

namespace NPackers {
#define _X_4(X) X, X, X, X
#define _X_8(X) _X_4(X), _X_4(X)
#define _X_16(X) _X_8(X), _X_8(X)
#define _X_32(X) _X_16(X), _X_16(X)
#define _X_64(X) _X_32(X), _X_32(X)
#define _X_128(X) _X_64(X), _X_64(X)

    const ui8 SkipTable[256] = {_X_128(1), _X_64(2), _X_32(3), _X_16(4), _X_8(5), _X_4(6), 7, 7, 8, 9};

#undef _X_4
#undef _X_8
#undef _X_16
#undef _X_32
#undef _X_64
#undef _X_128
}
