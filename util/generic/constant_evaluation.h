#pragma once

#include <util/system/compiler.h>

#include <type_traits>

/// Reports whether the current expression is evaluated at compile time.
///
/// Prefers the standard API, uses the compiler builtin in pre-C++20 modes, and
/// conservatively returns true when neither check is available.
constexpr bool IsConstantEvaluated() noexcept {
#if defined(__cpp_lib_is_constant_evaluated)
    return std::is_constant_evaluated();
#elif Y_HAS_BUILTIN(__builtin_is_constant_evaluated)
    return __builtin_is_constant_evaluated();
#else
    return true;
#endif
}
