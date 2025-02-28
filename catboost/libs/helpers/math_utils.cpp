#include "math_utils.h"

#include <util/system/cpu_id.h>
#include <util/system/platform.h>

// from library/cpp/fast_exp but they are not publicly defined
#if defined(_x86_64_) || defined(_i386_)
void FastExpInplaceAvx2(double* x, size_t count) noexcept;
void FastExpInplaceSse2(double* x, size_t count) noexcept;
#endif

namespace NCB {

    void FastExpWithInfInplace(double* x, size_t count) {
        #if defined(_x86_64_) || defined(_i386_)
            if (NX86::CachedHaveAVX() && NX86::CachedHaveAVX2()) {
                FastExpInplaceAvx2(x, count);
            } else if (NX86::CachedHaveSSE2()) {
                FastExpInplaceSse2(x, count);
            } else
        #endif
            {
                for (size_t i = 0; i < count; ++i) {
                    x[i] = std::exp(x[i]);
                }
            }
    }

}
