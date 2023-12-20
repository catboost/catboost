#pragma once

#if defined(_MSC_VER) && defined(__clang__)
    #define _compiler_clang_cl_
#elif defined(_MSC_VER)
    #define _compiler_msvc_
#elif defined(__clang__)
    #define _compiler_clang_
#elif defined(__GNUC__)
    #define _compiler_gcc_
#else
    #warning("Current compiler is not supported by " __FILE__)
#endif

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

// useful cross-platfrom definitions for compilers

/**
 * @def Y_FUNC_SIGNATURE
 *
 * Use this macro to get pretty function name (see example).
 *
 * @code
 * void Hi() {
 *     Cout << Y_FUNC_SIGNATURE << Endl;
 * }

 * template <typename T>
 * void Do() {
 *     Cout << Y_FUNC_SIGNATURE << Endl;
 * }

 * int main() {
 *    Hi();         // void Hi()
 *    Do<int>();    // void Do() [T = int]
 *    Do<TString>(); // void Do() [T = TString]
 * }
 * @endcode
 */
#if defined(__GNUC__)
    #define Y_FUNC_SIGNATURE __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
    #define Y_FUNC_SIGNATURE __FUNCSIG__
#else
    #define Y_FUNC_SIGNATURE ""
#endif

#ifdef __GNUC__
    #define Y_PRINTF_FORMAT(n, m) __attribute__((__format__(__printf__, n, m)))
#endif

#ifndef Y_PRINTF_FORMAT
    #define Y_PRINTF_FORMAT(n, m)
#endif

#if defined(__clang__)
    #define Y_NO_SANITIZE(...) __attribute__((no_sanitize(__VA_ARGS__)))
#endif

#if !defined(Y_NO_SANITIZE)
    #define Y_NO_SANITIZE(...)
#endif

/**
 * @def Y_DECLARE_UNUSED
 *
 * Macro is needed to silence compiler warning about unused entities (e.g. function or argument).
 *
 * @code
 * Y_DECLARE_UNUSED int FunctionUsedSolelyForDebugPurposes();
 * assert(FunctionUsedSolelyForDebugPurposes() == 42);
 *
 * void Foo(const int argumentUsedOnlyForDebugPurposes Y_DECLARE_UNUSED) {
 *     assert(argumentUsedOnlyForDebugPurposes == 42);
 *     // however you may as well omit `Y_DECLARE_UNUSED` and use `UNUSED` macro instead
 *     Y_UNUSED(argumentUsedOnlyForDebugPurposes);
 * }
 * @endcode
 */
#ifdef __GNUC__
    #define Y_DECLARE_UNUSED __attribute__((unused))
#endif

#ifndef Y_DECLARE_UNUSED
    #define Y_DECLARE_UNUSED
#endif

#if defined(__GNUC__)
    #define Y_LIKELY(Cond) __builtin_expect(!!(Cond), 1)
    #define Y_UNLIKELY(Cond) __builtin_expect(!!(Cond), 0)
    #define Y_PREFETCH_READ(Pointer, Priority) __builtin_prefetch((const void*)(Pointer), 0, Priority)
    #define Y_PREFETCH_WRITE(Pointer, Priority) __builtin_prefetch((const void*)(Pointer), 1, Priority)
#endif

/**
 * @def Y_FORCE_INLINE
 *
 * Macro to use in place of 'inline' in function declaration/definition to force
 * it to be inlined.
 */
#if !defined(Y_FORCE_INLINE)
    #if defined(CLANG_COVERAGE)
        #/* excessive __always_inline__ might significantly slow down compilation of an instrumented unit */
        #define Y_FORCE_INLINE inline
    #elif defined(_MSC_VER)
        #define Y_FORCE_INLINE __forceinline
    #elif defined(__GNUC__)
        #/* Clang also defines __GNUC__ (as 4) */
        #define Y_FORCE_INLINE inline __attribute__((__always_inline__))
    #else
        #define Y_FORCE_INLINE inline
    #endif
#endif

/**
 * @def Y_NO_INLINE
 *
 * Macro to use in place of 'inline' in function declaration/definition to
 * prevent it from being inlined.
 */
#if !defined(Y_NO_INLINE)
    #if defined(_MSC_VER)
        #define Y_NO_INLINE __declspec(noinline)
    #elif defined(__GNUC__) || defined(__INTEL_COMPILER)
        #/* Clang also defines __GNUC__ (as 4) */
        #define Y_NO_INLINE __attribute__((__noinline__))
    #else
        #define Y_NO_INLINE
    #endif
#endif

//to cheat compiler about strict aliasing or similar problems
#if defined(__GNUC__)
    #define Y_FAKE_READ(X)                  \
        do {                                \
            __asm__ __volatile__(""         \
                                 :          \
                                 : "m"(X)); \
        } while (0)

    #define Y_FAKE_WRITE(X)                  \
        do {                                 \
            __asm__ __volatile__(""          \
                                 : "=m"(X)); \
        } while (0)
#endif

#if !defined(Y_FAKE_READ)
    #define Y_FAKE_READ(X)
#endif

#if !defined(Y_FAKE_WRITE)
    #define Y_FAKE_WRITE(X)
#endif

#ifndef Y_PREFETCH_READ
    #define Y_PREFETCH_READ(Pointer, Priority) (void)(const void*)(Pointer), (void)Priority
#endif

#ifndef Y_PREFETCH_WRITE
    #define Y_PREFETCH_WRITE(Pointer, Priority) (void)(const void*)(Pointer), (void)Priority
#endif

#ifndef Y_LIKELY
    #define Y_LIKELY(Cond) (Cond)
    #define Y_UNLIKELY(Cond) (Cond)
#endif

#if defined(_compiler_clang_) || defined(_compiler_clang_cl_) || defined(_compiler_gcc_)
    #define Y_PACKED __attribute__((packed))
#else
    #define Y_PACKED
#endif

#if defined(__GNUC__)
    #define Y_WARN_UNUSED_RESULT __attribute__((warn_unused_result))
#endif

#ifndef Y_WARN_UNUSED_RESULT
    #define Y_WARN_UNUSED_RESULT
#endif

#if defined(__GNUC__)
    #define Y_HIDDEN __attribute__((visibility("hidden")))
#endif

#if !defined(Y_HIDDEN)
    #define Y_HIDDEN
#endif

#if defined(__GNUC__)
    #define Y_PUBLIC __attribute__((visibility("default")))
#endif

#if !defined(Y_PUBLIC)
    #define Y_PUBLIC
#endif

#if !defined(Y_UNUSED) && !defined(__cplusplus)
    #define Y_UNUSED(var) (void)(var)
#endif
#if !defined(Y_UNUSED) && defined(__cplusplus)
template <class... Types>
constexpr Y_FORCE_INLINE int Y_UNUSED(Types&&...) {
    return 0;
}
#endif

/**
 * @def Y_ASSUME
 *
 * Macro that tells the compiler that it can generate optimized code
 * as if the given expression will always evaluate true.
 * The behavior is undefined if it ever evaluates false.
 *
 * @code
 * // factored into a function so that it's testable
 * inline int Avg(int x, int y) {
 *     if (x >= 0 && y >= 0) {
 *         return (static_cast<unsigned>(x) + static_cast<unsigned>(y)) >> 1;
 *     } else {
 *         // a slower implementation
 *     }
 * }
 *
 * // we know that xs and ys are non-negative from domain knowledge,
 * // but we can't change the types of xs and ys because of API constrains
 * int Foo(const TVector<int>& xs, const TVector<int>& ys) {
 *     TVector<int> avgs;
 *     avgs.resize(xs.size());
 *     for (size_t i = 0; i < xs.size(); ++i) {
 *         auto x = xs[i];
 *         auto y = ys[i];
 *         Y_ASSUME(x >= 0);
 *         Y_ASSUME(y >= 0);
 *         xs[i] = Avg(x, y);
 *     }
 * }
 * @endcode
 */
#if defined(__GNUC__)
    #define Y_ASSUME(condition) ((condition) ? (void)0 : __builtin_unreachable())
#elif defined(_MSC_VER)
    #define Y_ASSUME(condition) __assume(condition)
#else
    #define Y_ASSUME(condition) Y_UNUSED(condition)
#endif

#ifdef __cplusplus
[[noreturn]]
#endif
Y_HIDDEN void
_YandexAbort();

/**
 * @def Y_UNREACHABLE
 *
 * Macro that marks the rest of the code branch unreachable.
 * The behavior is undefined if it's ever reached.
 *
 * @code
 * switch (i % 3) {
 * case 0:
 *     return foo;
 * case 1:
 *     return bar;
 * case 2:
 *     return baz;
 * default:
 *     Y_UNREACHABLE();
 * }
 * @endcode
 */
#if defined(__GNUC__)
    #define Y_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
    #define Y_UNREACHABLE() __assume(false)
#else
    #define Y_UNREACHABLE() _YandexAbort()
#endif

#if defined(undefined_sanitizer_enabled)
    #define _ubsan_enabled_
#endif

#ifdef __clang__

    #if __has_feature(thread_sanitizer)
        #define _tsan_enabled_
    #endif
    #if __has_feature(memory_sanitizer)
        #define _msan_enabled_
    #endif
    #if __has_feature(address_sanitizer)
        #define _asan_enabled_
    #endif

#else

    #if defined(thread_sanitizer_enabled) || defined(__SANITIZE_THREAD__)
        #define _tsan_enabled_
    #endif
    #if defined(memory_sanitizer_enabled)
        #define _msan_enabled_
    #endif
    #if defined(address_sanitizer_enabled) || defined(__SANITIZE_ADDRESS__)
        #define _asan_enabled_
    #endif

#endif

#if defined(_asan_enabled_) || defined(_msan_enabled_) || defined(_tsan_enabled_) || defined(_ubsan_enabled_)
    #define _san_enabled_
#endif

#if defined(_MSC_VER)
    #define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

#if defined(__GNUC__)
    #define Y_WEAK __attribute__((weak))
#else
    #define Y_WEAK
#endif

#if defined(__CUDACC_VER_MAJOR__)
    #define Y_CUDA_AT_LEAST(x, y) (__CUDACC_VER_MAJOR__ > x || (__CUDACC_VER_MAJOR__ == x && __CUDACC_VER_MINOR__ >= y))
#else
    #define Y_CUDA_AT_LEAST(x, y) 0
#endif

#if defined(__GNUC__)
    #define Y_COLD __attribute__((cold))
    #define Y_LEAF __attribute__((leaf))
    #define Y_WRAPPER __attribute__((artificial))
#else
    #define Y_COLD
    #define Y_LEAF
    #define Y_WRAPPER
#endif

/**
 * @def Y_PRAGMA
 *
 * Macro for use in other macros to define compiler pragma
 * See below for other usage examples
 *
 * @code
 * #if defined(__clang__) || defined(__GNUC__)
 * #define Y_PRAGMA_NO_WSHADOW \
 *     Y_PRAGMA("GCC diagnostic ignored \"-Wshadow\"")
 * #elif defined(_MSC_VER)
 * #define Y_PRAGMA_NO_WSHADOW \
 *     Y_PRAGMA("warning(disable:4456 4457")
 * #else
 * #define Y_PRAGMA_NO_WSHADOW
 * #endif
 * @endcode
 */
#if defined(__clang__) || defined(__GNUC__)
    #define Y_PRAGMA(x) _Pragma(x)
#elif defined(_MSC_VER)
    #define Y_PRAGMA(x) __pragma(x)
#else
    #define Y_PRAGMA(x)
#endif

/**
 * @def Y_PRAGMA_DIAGNOSTIC_PUSH
 *
 * Cross-compiler pragma to save diagnostic settings
 *
 * @see
 *     GCC: https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html
 *     MSVC: https://msdn.microsoft.com/en-us/library/2c8f766e.aspx
 *     Clang: https://clang.llvm.org/docs/UsersManual.html#controlling-diagnostics-via-pragmas
 *
 * @code
 * Y_PRAGMA_DIAGNOSTIC_PUSH
 * @endcode
 */
#if defined(__clang__) || defined(__GNUC__)
    #define Y_PRAGMA_DIAGNOSTIC_PUSH \
        Y_PRAGMA("GCC diagnostic push")
#elif defined(_MSC_VER)
    #define Y_PRAGMA_DIAGNOSTIC_PUSH \
        Y_PRAGMA(warning(push))
#else
    #define Y_PRAGMA_DIAGNOSTIC_PUSH
#endif

/**
 * @def Y_PRAGMA_DIAGNOSTIC_POP
 *
 * Cross-compiler pragma to restore diagnostic settings
 *
 * @see
 *     GCC: https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html
 *     MSVC: https://msdn.microsoft.com/en-us/library/2c8f766e.aspx
 *     Clang: https://clang.llvm.org/docs/UsersManual.html#controlling-diagnostics-via-pragmas
 *
 * @code
 * Y_PRAGMA_DIAGNOSTIC_POP
 * @endcode
 */
#if defined(__clang__) || defined(__GNUC__)
    #define Y_PRAGMA_DIAGNOSTIC_POP \
        Y_PRAGMA("GCC diagnostic pop")
#elif defined(_MSC_VER)
    #define Y_PRAGMA_DIAGNOSTIC_POP \
        Y_PRAGMA(warning(pop))
#else
    #define Y_PRAGMA_DIAGNOSTIC_POP
#endif

/**
 * @def Y_PRAGMA_NO_WSHADOW
 *
 * Cross-compiler pragma to disable warnings about shadowing variables
 *
 * @code
 * Y_PRAGMA_DIAGNOSTIC_PUSH
 * Y_PRAGMA_NO_WSHADOW
 *
 * // some code which use variable shadowing, e.g.:
 *
 * for (int i = 0; i < 100; ++i) {
 *   Use(i);
 *
 *   for (int i = 42; i < 100500; ++i) { // this i is shadowing previous i
 *       AnotherUse(i);
 *    }
 * }
 *
 * Y_PRAGMA_DIAGNOSTIC_POP
 * @endcode
 */
#if defined(__clang__) || defined(__GNUC__)
    #define Y_PRAGMA_NO_WSHADOW \
        Y_PRAGMA("GCC diagnostic ignored \"-Wshadow\"")
#elif defined(_MSC_VER)
    #define Y_PRAGMA_NO_WSHADOW \
        Y_PRAGMA(warning(disable : 4456 4457))
#else
    #define Y_PRAGMA_NO_WSHADOW
#endif

/**
 * @ def Y_PRAGMA_NO_UNUSED_FUNCTION
 *
 * Cross-compiler pragma to disable warnings about unused functions
 *
 * @see
 *     GCC: https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
 *     Clang: https://clang.llvm.org/docs/DiagnosticsReference.html#wunused-function
 *     MSVC: there is no such warning
 *
 * @code
 * Y_PRAGMA_DIAGNOSTIC_PUSH
 * Y_PRAGMA_NO_UNUSED_FUNCTION
 *
 * // some code which introduces a function which later will not be used, e.g.:
 *
 * void Foo() {
 * }
 *
 * int main() {
 *     return 0; // Foo() never called
 * }
 *
 * Y_PRAGMA_DIAGNOSTIC_POP
 * @endcode
 */
#if defined(__clang__) || defined(__GNUC__)
    #define Y_PRAGMA_NO_UNUSED_FUNCTION \
        Y_PRAGMA("GCC diagnostic ignored \"-Wunused-function\"")
#else
    #define Y_PRAGMA_NO_UNUSED_FUNCTION
#endif

/**
 * @ def Y_PRAGMA_NO_UNUSED_PARAMETER
 *
 * Cross-compiler pragma to disable warnings about unused function parameters
 *
 * @see
 *     GCC: https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
 *     Clang: https://clang.llvm.org/docs/DiagnosticsReference.html#wunused-parameter
 *     MSVC: https://msdn.microsoft.com/en-us/library/26kb9fy0.aspx
 *
 * @code
 * Y_PRAGMA_DIAGNOSTIC_PUSH
 * Y_PRAGMA_NO_UNUSED_PARAMETER
 *
 * // some code which introduces a function with unused parameter, e.g.:
 *
 * void foo(int a) {
 *     // a is not referenced
 * }
 *
 * int main() {
 *     foo(1);
 *     return 0;
 * }
 *
 * Y_PRAGMA_DIAGNOSTIC_POP
 * @endcode
 */
#if defined(__clang__) || defined(__GNUC__)
    #define Y_PRAGMA_NO_UNUSED_PARAMETER \
        Y_PRAGMA("GCC diagnostic ignored \"-Wunused-parameter\"")
#elif defined(_MSC_VER)
    #define Y_PRAGMA_NO_UNUSED_PARAMETER \
        Y_PRAGMA(warning(disable : 4100))
#else
    #define Y_PRAGMA_NO_UNUSED_PARAMETER
#endif

/**
 * @def Y_PRAGMA_NO_DEPRECATED
 *
 * Cross compiler pragma to disable warnings and errors about deprecated
 *
 * @see
 *     GCC: https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
 *     Clang: https://clang.llvm.org/docs/DiagnosticsReference.html#wdeprecated
 *     MSVC: https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-3-c4996?view=vs-2017
 *
 * @code
 * Y_PRAGMA_DIAGNOSTIC_PUSH
 * Y_PRAGMA_NO_DEPRECATED
 *
 * [deprecated] void foo() {
 *     // ...
 * }
 *
 * int main() {
 *     foo();
 *     return 0;
 * }
 *
 * Y_PRAGMA_DIAGNOSTIC_POP
 * @endcode
 */
#if defined(__clang__) || defined(__GNUC__)
    #define Y_PRAGMA_NO_DEPRECATED \
        Y_PRAGMA("GCC diagnostic ignored \"-Wdeprecated\"")
#elif defined(_MSC_VER)
    #define Y_PRAGMA_NO_DEPRECATED \
        Y_PRAGMA(warning(disable : 4996))
#else
    #define Y_PRAGMA_NO_DEPRECATED
#endif

// Memory sanitizer sometimes doesn't correctly set parameter shadow of constant functions.
#if (defined(__clang__) || defined(__GNUC__)) && !defined(_msan_enabled_)
    /**
 * @def Y_CONST_FUNCTION
   methods and functions, marked with this method are promised to:
     1. do not have side effects
     2. this method do not read global memory
   NOTE: this attribute can't be set for methods that depend on data, pointed by this
   this allow compilers to do hard optimization of that functions
   NOTE: in common case this attribute can't be set if method have pointer-arguments
   NOTE: as result there no any reason to discard result of such method
*/
    #define Y_CONST_FUNCTION [[gnu::const]]
#endif

#if !defined(Y_CONST_FUNCTION)
    #define Y_CONST_FUNCTION
#endif

#if defined(__clang__) || defined(__GNUC__)
    /**
 * @def Y_PURE_FUNCTION
   methods and functions, marked with this method are promised to:
     1. do not have side effects
     2. result will be the same if no global memory changed
   this allow compilers to do hard optimization of that functions
   NOTE: as result there no any reason to discard result of such method
*/
    #define Y_PURE_FUNCTION [[gnu::pure]]
#endif

#if !defined(Y_PURE_FUNCTION)
    #define Y_PURE_FUNCTION
#endif

/**
 * @ def Y_HAVE_INT128
 *
 * Defined when the compiler supports __int128 extension
 *
 * @code
 *
 * #if defined(Y_HAVE_INT128)
 *     __int128 myVeryBigInt = 12345678901234567890;
 * #endif
 *
 * @endcode
 */
#if defined(__SIZEOF_INT128__)
    #define Y_HAVE_INT128 1
#endif

#if defined(__clang__) && (!defined(__CUDACC__) || Y_CUDA_AT_LEAST(11, 0))
    #define Y_REINITIALIZES_OBJECT [[clang::reinitializes]]
#else
    #define Y_REINITIALIZES_OBJECT
#endif

// Use at the end of macros declaration. It allows macros usage only with semicolon at the end.
// It prevents from warnings for extra semicolons when building with flag `-Wextra-semi`.
#define Y_SEMICOLON_GUARD static_assert(true, "")

#ifdef __cplusplus

void UseCharPointerImpl(volatile const char*);

template <typename T>
Y_FORCE_INLINE void DoNotOptimizeAway(T&& datum) {
    #if defined(_MSC_VER)
    UseCharPointerImpl(&reinterpret_cast<volatile const char&>(datum));
    _ReadWriteBarrier();
    #elif defined(__GNUC__) && defined(_x86_)
    asm volatile(""
                 :
                 : "X"(datum));
    #else
    Y_FAKE_READ(datum);
    #endif
}

/**
 * The usage for `const T&` is prohibited.
 * The compiler assume that a constant reference, even though escaped via asm volatile, is unchanged.
 * The const-ref interface is deleted to discourage new uses of it, as subtle compiler optimizations (invariant hoisting, etc.) can occur.
 * For more details see https://github.com/google/benchmark/pull/1493.
 */
template <typename T>
Y_FORCE_INLINE void DoNotOptimizeAway(const T&) = delete;

    /**
     * Use this macro to prevent unused variables elimination.
     */
    #define Y_DO_NOT_OPTIMIZE_AWAY(X) ::DoNotOptimizeAway(X)

#endif

/**
 * @def Y_LIFETIME_BOUND
 *
 * The attribute on a function parameter can be used to tell the compiler
 * that function return value may refer that parameter.
 * The compiler may produce compile-time warning if it is able to detect that
 * an object or reference refers to another object with a shorter lifetime.
 */
#if defined(__clang__) && defined(__cplusplus) && defined(__has_cpp_attribute)
    #if defined(__CUDACC__) && !Y_CUDA_AT_LEAST(11, 0)
        #define Y_LIFETIME_BOUND
    #elif __has_cpp_attribute(clang::lifetimebound)
        #define Y_LIFETIME_BOUND [[clang::lifetimebound]]
    #else
        #define Y_LIFETIME_BOUND
    #endif
#else
    #define Y_LIFETIME_BOUND
#endif

/**
 * @def Y_HAVE_ATTRIBUTE
 *
 * A function-like feature checking macro that is a wrapper around
 * `__has_attribute`, which is defined by GCC 5+ and Clang and evaluates to a
 * nonzero constant integer if the attribute is supported or 0 if not.
 *
 * It evaluates to zero if `__has_attribute` is not defined by the compiler.
 *
 * @see
 *     GCC: https://gcc.gnu.org/gcc-5/changes.html
 *     Clang: https://clang.llvm.org/docs/LanguageExtensions.html
 */
#ifdef __has_attribute
    #define Y_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
    #define Y_HAVE_ATTRIBUTE(x) 0
#endif

/**
 * @def Y_RETURNS_NONNULL
 *
 * The returns_nonnull attribute specifies that the function return value should
 * be a non-null pointer. It lets the compiler optimize callers based on
 * the knowledge that the return value will never be null.
 *
 * @see
 *    GCC: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-returns_005fnonnull-function-attribute
 *    Clang: https://clang.llvm.org/docs/AttributeReference.html#returns-nonnull
 *
 * @code
 * Y_RETURNS_NONNULL extern void* mymalloc(size_t len);
 * @endcode
 */
#if Y_HAVE_ATTRIBUTE(returns_nonnull)
    #define Y_RETURNS_NONNULL __attribute__((returns_nonnull))
#else
    #define Y_RETURNS_NONNULL
#endif
