#include "fast_exp.h"

#include <util/system/cpu_id.h>
#include <util/system/types.h>
#include <util/generic/singleton.h>

#include <cmath>

#if defined(_x86_64_) || defined(_i386_)
void FastExpInplaceAvx2(double* x, size_t count) noexcept;
void FastExpInplaceSse2(double* x, size_t count) noexcept;
#endif

namespace {
    union TDoubleInt {
        double FVal;
        ui64 IVal;
    };

    struct TTable {
        inline TTable() {
            for (int i = 0; i < 65536; ++i) {
                TDoubleInt x;
                x.IVal = ((ui64)i) << 48;
                ExpTable[i] = std::exp(x.FVal);
            }
        }

        double ExpTable[65536];
    };
}

double fast_exp(double x) {
    double arg = x;
    double res = 1;

    const auto& expTable = HugeSingletonWithPriority<TTable, 0>()->ExpTable;

    for (int iter = 0; iter < 4; ++iter) {
        TDoubleInt dd;
        dd.FVal = arg;
        double chk = expTable[dd.IVal >> 48];
        dd.IVal &= 0xffff000000000000ll;
        res *= chk;
        arg -= dd.FVal;
    }

    return res * (1 + arg);
}

void FastExpInplace(double* x, size_t count) {
#if defined(_x86_64_) || defined(_i386_)
    if (NX86::CachedHaveAVX() && NX86::CachedHaveAVX2()) {
        FastExpInplaceAvx2(x, count);
    } else if (NX86::CachedHaveSSE2()) {
        FastExpInplaceSse2(x, count);
    } else
#endif
    {
        for (size_t i = 0; i < count; ++i) {
            x[i] = fast_exp(x[i]);
        }
    }
}
