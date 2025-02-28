#pragma once

#include <util/system/types.h>

#include <cmath>


namespace NCB {

    /* TODO(akhropov): replace with fast implementation,
     *  some ideas: https://stackoverflow.com/questions/3272424/compute-fast-log-base-2-ceiling
     *  util/generic/bitops/CeilLog2 won't work due to 'CeilLog2(1)=1'
     */
    inline ui32 IntLog2(ui32 values) {
        return (ui32)ceil(log2(values));
    }

    // nan == nan is true here in contrast with the standard comparison operator
    template <class T>
    inline bool EqualWithNans(T lhs, T rhs) {
        if (std::isnan(lhs)) {
            return std::isnan(rhs);
        }
        return lhs == rhs;
    }

    inline double Logit(double x) {
        return -log(1 / x - 1);
    }

    // If SSE2 is not enabled
    // uses std::exp instead of fast_exp from library/cpp/fast_exp because fast_exp does not work with +/-inf.
    void FastExpWithInfInplace(double* x, size_t count);
}

