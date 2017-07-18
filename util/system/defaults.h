#pragma once

#include "platform.h"

#if defined _unix_
#define LOCSLASH_C '/'
#define LOCSLASH_S "/"
#else
#define LOCSLASH_C '\\'
#define LOCSLASH_S "\\"
#endif // _unix_

#if defined(__INTEL_COMPILER) && defined(__cplusplus)
#include <new>
#endif

// low and high parts of integers
#if !defined(_win_)
#include <sys/param.h>
#endif

#if defined(BSD)
#include <machine/endian.h>
#if (BYTE_ORDER == LITTLE_ENDIAN)
#define _little_endian_
#elif (BYTE_ORDER == BIG_ENDIAN)
#define _big_endian_
#else
#error unknown endian not supported
#endif
#elif (defined(_sun_) && !defined(__i386__)) || defined(_hpux_) || defined(WHATEVER_THAT_HAS_BIG_ENDIAN)
#define _big_endian_
#else
#define _little_endian_
#endif

// alignment
#if (defined(_sun_) && !defined(__i386__)) || defined(_hpux_) || defined(__alpha__) || defined(__ia64__) || defined(WHATEVER_THAT_NEEDS_ALIGNING_QUADS)
#define _must_align8_
#endif

#if (defined(_sun_) && !defined(__i386__)) || defined(_hpux_) || defined(__alpha__) || defined(__ia64__) || defined(WHATEVER_THAT_NEEDS_ALIGNING_LONGS)
#define _must_align4_
#endif

#if (defined(_sun_) && !defined(__i386__)) || defined(_hpux_) || defined(__alpha__) || defined(__ia64__) || defined(WHATEVER_THAT_NEEDS_ALIGNING_SHORTS)
#define _must_align2_
#endif

#if defined(__GNUC__)
#define alias_hack __attribute__((__may_alias__))
#endif

#ifndef alias_hack
#define alias_hack
#endif

#include "types.h"

typedef ui16 alias_hack ui16a;
typedef ui32 alias_hack ui32a;
typedef ui64 alias_hack ui64a;

#if defined(__cplusplus)
#if defined(_big_endian_)
union u_u16 {
    ui16a v;
    struct {
        ui8 hi8, lo8;
    } u;
} alias_hack;
union u_u32 {
    ui32a v;
    float alias_hack f;
    struct {
        u_u16 hi16, lo16;
    } u;
} alias_hack;
union u_u64 {
    ui64a v;
    double alias_hack f;
    struct {
        u_u32 hi32, lo32;
    } u;
} alias_hack;
#else /* _little_endian_ */
union u_u16 {
    ui16a v;
    struct {
        ui8 lo8, hi8;
    } alias_hack u;
} alias_hack;
union u_u32 {
    ui32a v;
    float alias_hack f;
    struct {
        u_u16 lo16, hi16;
    } u;
} alias_hack;
union u_u64 {
    ui64a v;
    double alias_hack f;
    struct {
        u_u32 lo32, hi32;
    } u;
} alias_hack;
#endif
#endif

#ifdef CHECK_LO_HI_MACRO_USAGE

inline void check_64(const ui64&) {
}
inline void check_64(const i64&) {
}
inline void check_64(const double&) {
}
inline void check_32(const ui32&) {
}
inline void check_32(const i32&) {
}
inline void check_32(const float&) {
}
inline void check_16(const ui16&) {
}
inline void check_16(const i16&) {
}

#define LO_32(x) (check_64(x), (ui32&)(*(u_u64*)&x).u.lo32.v)
#define HI_32(x) (check_64(x), (ui32&)(*(u_u64*)&x).u.hi32.v)
#define LO_16(x) (check_32(x), (ui16&)(*(u_u32*)&x).u.lo16.v)
#define HI_16(x) (check_32(x), (ui16&)(*(u_u32*)&x).u.hi16.v)
#define LO_8(x) (check_16(x), (*(u_u16*)&x).u.lo8)
#define HI_8(x) (check_16(x), (*(u_u16*)&x).u.hi8)
#define LO_8_LO_16(x) (check_32(x), (*(u_u32*)&x).u.lo16.u.lo8)
#define HI_8_LO_16(x) (check_32(x), (*(u_u32*)&x).u.lo16.u.hi8)

#else

#define LO_32(x) ((ui32&)(*(u_u64*)&x).u.lo32.v)
#define HI_32(x) ((ui32&)(*(u_u64*)&x).u.hi32.v)
#define LO_16(x) ((ui16&)(*(u_u32*)&x).u.lo16.v)
#define HI_16(x) ((ui16&)(*(u_u32*)&x).u.hi16.v)
#define LO_8(x) (*(u_u16*)&x).u.lo8
#define HI_8(x) (*(u_u16*)&x).u.hi8
#define LO_8_LO_16(x) (*(u_u32*)&x).u.lo16.u.lo8
#define HI_8_LO_16(x) (*(u_u32*)&x).u.lo16.u.hi8

#endif // CHECK_LO_HI_MACRO_USAGE

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
#define PRAGMA(x) _Pragma(#x)
#define RCSID(idstr) PRAGMA(comment(exestr, idstr))
#else
#define RCSID(idstr) static const char rcsid[] = idstr
#endif

#include "compiler.h"

/// Deprecated. Use TNonCopyable instead (util/generic/noncopyable.h)
#define Y_DISABLE_COPY(aClass) \
private:                       \
    aClass(const aClass&);     \
    aClass& operator=(const aClass&)

#ifdef _win_
#include <malloc.h>
#elif defined(_sun_)
#include <alloca.h>
#endif

#ifdef NDEBUG
#define Y_IF_DEBUG(X)
#else
#define Y_IF_DEBUG(X) X
#endif

/**
 * @def Y_ARRAY_SIZE
 *
 * This macro is needed to get number of elements in a statically allocated fixed size array. The
 * expression is a compile-time constant and therefore can be used in compile time computations.
 *
 * @code
 * enum ENumbers {
 *     EN_ONE,
 *     EN_TWO,
 *     EN_SIZE
 * }
 *
 * const char* NAMES[] = {
 *     "one",
 *     "two"
 * }
 *
 * static_assert(Y_ARRAY_SIZE(NAMES) == EN_SIZE, "you should define `NAME` for each enumeration");
 * @endcode
 *
 * This macro also catches type errors. If you see a compiler error like "warning: division by zero
 * is undefined" when using `Y_ARRAY_SIZE` then you are probably giving it a pointer.
 *
 * Since all of our code is expected to work on a 64 bit platform where pointers are 8 bytes we may
 * falsefully accept pointers to types of sizes that are divisors of 8 (1, 2, 4 and 8).
 */
#if defined(__cplusplus)
#include <util/generic/array_size.h>
#else
#undef Y_ARRAY_SIZE
#define Y_ARRAY_SIZE(arr) \
    ((sizeof(arr) / sizeof((arr)[0])) / static_cast<size_t>(!(sizeof(arr) % sizeof((arr)[0]))))
#endif

#undef Y_ARRAY_BEGIN
#define Y_ARRAY_BEGIN(arr) (arr)

#undef Y_ARRAY_END
#define Y_ARRAY_END(arr) ((arr) + Y_ARRAY_SIZE(arr))

/**
 * Concatenates two symbols, even if one of them is itself a macro.
 */
#define Y_CAT(X, Y) Y_CAT_I(X, Y)
#define Y_CAT_I(X, Y) Y_CAT_II(X, Y)
#define Y_CAT_II(X, Y) X##Y

#define Y_STRINGIZE(X) UTIL_PRIVATE_STRINGIZE_AUX(X)
#define UTIL_PRIVATE_STRINGIZE_AUX(X) #X

#if defined(__COUNTER__)
#define Y_GENERATE_UNIQUE_ID(N) Y_CAT(N, __COUNTER__)
#endif

#if !defined(Y_GENERATE_UNIQUE_ID)
#define Y_GENERATE_UNIQUE_ID(N) Y_CAT(N, __LINE__)
#endif

#define NPOS ((size_t)-1)
