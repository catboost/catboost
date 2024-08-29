//
// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -----------------------------------------------------------------------------
// File: optimization.h
// -----------------------------------------------------------------------------
//
// This header file defines portable macros for performance optimization.
//
// This header is included in both C++ code and legacy C code and thus must
// remain compatible with both C and C++. C compatibility will be removed if
// the legacy code is removed or converted to C++. Do not include this header in
// new code that requires C compatibility or assume C compatibility will remain
// indefinitely.

#ifndef Y_ABSL_BASE_OPTIMIZATION_H_
#define Y_ABSL_BASE_OPTIMIZATION_H_

#include <assert.h>

#ifdef __cplusplus
// Included for std::unreachable()
#include <utility>
#endif  // __cplusplus

#include "y_absl/base/config.h"
#include "y_absl/base/options.h"

// Y_ABSL_BLOCK_TAIL_CALL_OPTIMIZATION
//
// Instructs the compiler to avoid optimizing tail-call recursion. This macro is
// useful when you wish to preserve the existing function order within a stack
// trace for logging, debugging, or profiling purposes.
//
// Example:
//
//   int f() {
//     int result = g();
//     Y_ABSL_BLOCK_TAIL_CALL_OPTIMIZATION();
//     return result;
//   }
#if defined(__pnacl__)
#define Y_ABSL_BLOCK_TAIL_CALL_OPTIMIZATION() if (volatile int x = 0) { (void)x; }
#elif defined(__clang__)
// Clang will not tail call given inline volatile assembly.
#define Y_ABSL_BLOCK_TAIL_CALL_OPTIMIZATION() __asm__ __volatile__("")
#elif defined(__GNUC__)
// GCC will not tail call given inline volatile assembly.
#define Y_ABSL_BLOCK_TAIL_CALL_OPTIMIZATION() __asm__ __volatile__("")
#elif defined(_MSC_VER)
#include <intrin.h>
// The __nop() intrinsic blocks the optimisation.
#define Y_ABSL_BLOCK_TAIL_CALL_OPTIMIZATION() __nop()
#else
#define Y_ABSL_BLOCK_TAIL_CALL_OPTIMIZATION() if (volatile int x = 0) { (void)x; }
#endif

// Y_ABSL_CACHELINE_SIZE
//
// Explicitly defines the size of the L1 cache for purposes of alignment.
// Setting the cacheline size allows you to specify that certain objects be
// aligned on a cacheline boundary with `Y_ABSL_CACHELINE_ALIGNED` declarations.
// (See below.)
//
// NOTE: this macro should be replaced with the following C++17 features, when
// those are generally available:
//
//   * `std::hardware_constructive_interference_size`
//   * `std::hardware_destructive_interference_size`
//
// See http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0154r1.html
// for more information.
#if defined(__GNUC__)
// Cache line alignment
#if defined(__i386__) || defined(__x86_64__)
#define Y_ABSL_CACHELINE_SIZE 64
#elif defined(__powerpc64__)
#define Y_ABSL_CACHELINE_SIZE 128
#elif defined(__aarch64__)
// We would need to read special register ctr_el0 to find out L1 dcache size.
// This value is a good estimate based on a real aarch64 machine.
#define Y_ABSL_CACHELINE_SIZE 64
#elif defined(__arm__)
// Cache line sizes for ARM: These values are not strictly correct since
// cache line sizes depend on implementations, not architectures.  There
// are even implementations with cache line sizes configurable at boot
// time.
#if defined(__ARM_ARCH_5T__)
#define Y_ABSL_CACHELINE_SIZE 32
#elif defined(__ARM_ARCH_7A__)
#define Y_ABSL_CACHELINE_SIZE 64
#endif
#endif
#endif

#ifndef Y_ABSL_CACHELINE_SIZE
// A reasonable default guess.  Note that overestimates tend to waste more
// space, while underestimates tend to waste more time.
#define Y_ABSL_CACHELINE_SIZE 64
#endif

// Y_ABSL_CACHELINE_ALIGNED
//
// Indicates that the declared object be cache aligned using
// `Y_ABSL_CACHELINE_SIZE` (see above). Cacheline aligning objects allows you to
// load a set of related objects in the L1 cache for performance improvements.
// Cacheline aligning objects properly allows constructive memory sharing and
// prevents destructive (or "false") memory sharing.
//
// NOTE: callers should replace uses of this macro with `alignas()` using
// `std::hardware_constructive_interference_size` and/or
// `std::hardware_destructive_interference_size` when C++17 becomes available to
// them.
//
// See http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0154r1.html
// for more information.
//
// On some compilers, `Y_ABSL_CACHELINE_ALIGNED` expands to an `__attribute__`
// or `__declspec` attribute. For compilers where this is not known to work,
// the macro expands to nothing.
//
// No further guarantees are made here. The result of applying the macro
// to variables and types is always implementation-defined.
//
// WARNING: It is easy to use this attribute incorrectly, even to the point
// of causing bugs that are difficult to diagnose, crash, etc. It does not
// of itself guarantee that objects are aligned to a cache line.
//
// NOTE: Some compilers are picky about the locations of annotations such as
// this attribute, so prefer to put it at the beginning of your declaration.
// For example,
//
//   Y_ABSL_CACHELINE_ALIGNED static Foo* foo = ...
//
//   class Y_ABSL_CACHELINE_ALIGNED Bar { ...
//
// Recommendations:
//
// 1) Consult compiler documentation; this comment is not kept in sync as
//    toolchains evolve.
// 2) Verify your use has the intended effect. This often requires inspecting
//    the generated machine code.
// 3) Prefer applying this attribute to individual variables. Avoid
//    applying it to types. This tends to localize the effect.
#if defined(__clang__) || defined(__GNUC__)
#define Y_ABSL_CACHELINE_ALIGNED __attribute__((aligned(Y_ABSL_CACHELINE_SIZE)))
#elif defined(_MSC_VER)
#define Y_ABSL_CACHELINE_ALIGNED __declspec(align(Y_ABSL_CACHELINE_SIZE))
#else
#define Y_ABSL_CACHELINE_ALIGNED
#endif

// Y_ABSL_PREDICT_TRUE, Y_ABSL_PREDICT_FALSE
//
// Enables the compiler to prioritize compilation using static analysis for
// likely paths within a boolean branch.
//
// Example:
//
//   if (Y_ABSL_PREDICT_TRUE(expression)) {
//     return result;                        // Faster if more likely
//   } else {
//     return 0;
//   }
//
// Compilers can use the information that a certain branch is not likely to be
// taken (for instance, a CHECK failure) to optimize for the common case in
// the absence of better information (ie. compiling gcc with `-fprofile-arcs`).
//
// Recommendation: Modern CPUs dynamically predict branch execution paths,
// typically with accuracy greater than 97%. As a result, annotating every
// branch in a codebase is likely counterproductive; however, annotating
// specific branches that are both hot and consistently mispredicted is likely
// to yield performance improvements.
#if Y_ABSL_HAVE_BUILTIN(__builtin_expect) || \
    (defined(__GNUC__) && !defined(__clang__))
#define Y_ABSL_PREDICT_FALSE(x) (__builtin_expect(false || (x), false))
#define Y_ABSL_PREDICT_TRUE(x) (__builtin_expect(false || (x), true))
#else
#define Y_ABSL_PREDICT_FALSE(x) (x)
#define Y_ABSL_PREDICT_TRUE(x) (x)
#endif

// `Y_ABSL_INTERNAL_IMMEDIATE_ABORT_IMPL()` aborts the program in the fastest
// possible way, with no attempt at logging. One use is to implement hardening
// aborts with Y_ABSL_OPTION_HARDENED.  Since this is an internal symbol, it
// should not be used directly outside of Abseil.
#if Y_ABSL_HAVE_BUILTIN(__builtin_trap) || \
    (defined(__GNUC__) && !defined(__clang__))
#define Y_ABSL_INTERNAL_IMMEDIATE_ABORT_IMPL() __builtin_trap()
#else
#define Y_ABSL_INTERNAL_IMMEDIATE_ABORT_IMPL() abort()
#endif

// `Y_ABSL_INTERNAL_UNREACHABLE_IMPL()` is the platform specific directive to
// indicate that a statement is unreachable, and to allow the compiler to
// optimize accordingly. Clients should use `Y_ABSL_UNREACHABLE()`, which is
// defined below.
#if defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#define Y_ABSL_INTERNAL_UNREACHABLE_IMPL() std::unreachable()
#elif defined(__GNUC__) || Y_ABSL_HAVE_BUILTIN(__builtin_unreachable)
#define Y_ABSL_INTERNAL_UNREACHABLE_IMPL() __builtin_unreachable()
#elif Y_ABSL_HAVE_BUILTIN(__builtin_assume)
#define Y_ABSL_INTERNAL_UNREACHABLE_IMPL() __builtin_assume(false)
#elif defined(_MSC_VER)
#define Y_ABSL_INTERNAL_UNREACHABLE_IMPL() __assume(false)
#else
#define Y_ABSL_INTERNAL_UNREACHABLE_IMPL()
#endif

// `Y_ABSL_UNREACHABLE()` is an unreachable statement.  A program which reaches
// one has undefined behavior, and the compiler may optimize accordingly.
#if Y_ABSL_OPTION_HARDENED == 1 && defined(NDEBUG)
// Abort in hardened mode to avoid dangerous undefined behavior.
#define Y_ABSL_UNREACHABLE()                \
  do {                                    \
    Y_ABSL_INTERNAL_IMMEDIATE_ABORT_IMPL(); \
    Y_ABSL_INTERNAL_UNREACHABLE_IMPL();     \
  } while (false)
#else
// The assert only fires in debug mode to aid in debugging.
// When NDEBUG is defined, reaching Y_ABSL_UNREACHABLE() is undefined behavior.
#define Y_ABSL_UNREACHABLE()                       \
  do {                                           \
    /* NOLINTNEXTLINE: misc-static-assert */     \
    assert(false && "Y_ABSL_UNREACHABLE reached"); \
    Y_ABSL_INTERNAL_UNREACHABLE_IMPL();            \
  } while (false)
#endif

// Y_ABSL_ASSUME(cond)
//
// Informs the compiler that a condition is always true and that it can assume
// it to be true for optimization purposes.
//
// WARNING: If the condition is false, the program can produce undefined and
// potentially dangerous behavior.
//
// In !NDEBUG mode, the condition is checked with an assert().
//
// NOTE: The expression must not have side effects, as it may only be evaluated
// in some compilation modes and not others. Some compilers may issue a warning
// if the compiler cannot prove the expression has no side effects. For example,
// the expression should not use a function call since the compiler cannot prove
// that a function call does not have side effects.
//
// Example:
//
//   int x = ...;
//   Y_ABSL_ASSUME(x >= 0);
//   // The compiler can optimize the division to a simple right shift using the
//   // assumption specified above.
//   int y = x / 16;
//
#if !defined(NDEBUG)
#define Y_ABSL_ASSUME(cond) assert(cond)
#elif Y_ABSL_HAVE_BUILTIN(__builtin_assume)
#define Y_ABSL_ASSUME(cond) __builtin_assume(cond)
#elif defined(_MSC_VER)
#define Y_ABSL_ASSUME(cond) __assume(cond)
#elif defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#define Y_ABSL_ASSUME(cond)            \
  do {                               \
    if (!(cond)) std::unreachable(); \
  } while (false)
#elif defined(__GNUC__) || Y_ABSL_HAVE_BUILTIN(__builtin_unreachable)
#define Y_ABSL_ASSUME(cond)                 \
  do {                                    \
    if (!(cond)) __builtin_unreachable(); \
  } while (false)
#else
#define Y_ABSL_ASSUME(cond)               \
  do {                                  \
    static_cast<void>(false && (cond)); \
  } while (false)
#endif

// Y_ABSL_INTERNAL_UNIQUE_SMALL_NAME(cond)
// This macro forces small unique name on a static file level symbols like
// static local variables or static functions. This is intended to be used in
// macro definitions to optimize the cost of generated code. Do NOT use it on
// symbols exported from translation unit since it may cause a link time
// conflict.
//
// Example:
//
// #define MY_MACRO(txt)
// namespace {
//  char VeryVeryLongVarName[] Y_ABSL_INTERNAL_UNIQUE_SMALL_NAME() = txt;
//  const char* VeryVeryLongFuncName() Y_ABSL_INTERNAL_UNIQUE_SMALL_NAME();
//  const char* VeryVeryLongFuncName() { return txt; }
// }
//

#if defined(__GNUC__)
#define Y_ABSL_INTERNAL_UNIQUE_SMALL_NAME2(x) #x
#define Y_ABSL_INTERNAL_UNIQUE_SMALL_NAME1(x) Y_ABSL_INTERNAL_UNIQUE_SMALL_NAME2(x)
#define Y_ABSL_INTERNAL_UNIQUE_SMALL_NAME() \
  asm(Y_ABSL_INTERNAL_UNIQUE_SMALL_NAME1(.y_absl.__COUNTER__))
#else
#define Y_ABSL_INTERNAL_UNIQUE_SMALL_NAME()
#endif

#endif  // Y_ABSL_BASE_OPTIMIZATION_H_
