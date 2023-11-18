#ifndef CONTOURPY_UTIL_H
#define CONTOURPY_UTIL_H

#include "common.h"

namespace contourpy {

class Util
{
public:
    static void ensure_nan_loaded();

    static index_t get_max_threads();

    // This is the NaN used internally and returned to calling functions. The value is taken from
    // numpy rather than the standard C++ approach so that it is guaranteed to work with
    // numpy.isnan(). The value is actually the same for many platforms, but this approach
    // guarantees it works for all platforms that numpy supports.
    //
    // ensure_nan_loaded() must be called before this value is read.
    static double nan;

private:
    static bool _nan_loaded;
};

} // namespace contourpy

#endif // CONTOURPY_UTIL_H
