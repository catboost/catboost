#include <cxxabi.h>

/**
 * libc++ expects std::uncaught_exceptions() to be provided by C++ runtime library.
 *
 * GCC versions prior to GCC 6 did not provide this function yet, but it can be
 * implemented using its API.
 *
 * This implementation should cover ubuntu-12, ubuntu-14, ubuntu-16.
 */

namespace std {
    int uncaught_exceptions() noexcept {
        const auto* globals{__cxxabiv1::__cxa_get_globals()};
        return static_cast<int>(globals->uncaughtExceptions);
    }
}
