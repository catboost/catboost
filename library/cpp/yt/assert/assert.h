#pragma once

#include <util/system/compiler.h>
#include <util/system/src_root.h>

#include <util/generic/strbuf.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

[[noreturn]]
void AssertTrapImpl(
    TStringBuf trapType,
    TStringBuf expr,
    TStringBuf file,
    int line,
    TStringBuf function);

} // namespace NDetail

#ifdef __GNUC__
    #define YT_BUILTIN_TRAP()  __builtin_trap()
#else
    #define YT_BUILTIN_TRAP()  std::terminate()
#endif

#define YT_ASSERT_TRAP(trapType, expr) \
    ::NYT::NDetail::AssertTrapImpl(TStringBuf(trapType), TStringBuf(expr), __SOURCE_FILE_IMPL__.As<TStringBuf>(), __LINE__, TStringBuf(__FUNCTION__)); \
    Y_UNREACHABLE() \

#ifdef NDEBUG
    #define YT_ASSERT(expr) \
        do { \
            if (false) { \
                (void) (expr); \
            } \
        } while (false)
#else
    #define YT_ASSERT(expr) \
        do { \
            if (Y_UNLIKELY(!(expr))) { \
                YT_ASSERT_TRAP("YT_ASSERT", #expr); \
            } \
        } while (false)
#endif

//! Same as |YT_ASSERT| but evaluates and checks the expression in both release and debug mode.
#define YT_VERIFY(expr) \
    do { \
        if (Y_UNLIKELY(!(expr))) { \
            YT_ASSERT_TRAP("YT_VERIFY", #expr); \
        } \
    } while (false)

//! Behaves as |YT_ASSERT| in debug mode and as |Y_ASSUME| in release.
#ifdef NDEBUG
    #define YT_ASSUME(expr) Y_ASSUME(expr)
#else
    #define YT_ASSUME(expr) YT_ASSERT(expr)
#endif

//! Behaves as |YT_ASSERT(false)| in debug mode and as |Y_UNREACHABLE| in release.
#ifdef NDEBUG
    #define YT_UNREACHABLE() Y_UNREACHABLE()
#else
    #define YT_UNREACHABLE() YT_ASSERT(false)
#endif

//! Fatal error code marker. Abnormally terminates the current process.
#ifdef YT_COMPILING_UDF
    #define YT_ABORT() __YT_BUILTIN_ABORT()
#else
    #define YT_ABORT() \
        do { \
            YT_ASSERT_TRAP("YT_ABORT", ""); \
        } while (false)
#endif

//! Unimplemented code marker. Abnormally terminates the current process.
#define YT_UNIMPLEMENTED() \
    do { \
        YT_ASSERT_TRAP("YT_UNIMPLEMENTED", ""); \
    } while (false)

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
