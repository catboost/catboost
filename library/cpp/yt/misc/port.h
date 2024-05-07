#pragma once

#include <util/system/platform.h>

// Check platform bitness.
#if !defined(_64_)
    #error YT requires 64-bit platform
#endif

// This define enables tracking of reference-counted objects to provide
// various insightful information on memory usage and object creation patterns.
#define YT_ENABLE_REF_COUNTED_TRACKING

// This define enables logging with TRACE level. You can still disable trace logging
// for particular TU by discarding this macro identifier.
#define YT_ENABLE_TRACE_LOGGING

// This define should be used to compile YT with vanilla protobuf instead of patched one.
// #define YT_USE_VANILLA_PROTOBUF

#ifndef NDEBUG
    // This define enables thread affinity check -- a user-defined verification ensuring
    // that some functions are called from particular threads.
    #define YT_ENABLE_THREAD_AFFINITY_CHECK

    // This define enables tracking of BIND callbacks location.
    #define YT_ENABLE_BIND_LOCATION_TRACKING

    // This define enables checking that all required protobuf fields are present
    // during serialization.
    #define YT_VALIDATE_REQUIRED_PROTO_FIELDS

    // Detects deadlocks caused by recursive acquisitions of (non-recursive) spin locks.
    #define YT_ENABLE_SPIN_LOCK_OWNERSHIP_TRACKING
#endif

// Configure SSE usage.
#ifdef SSE42_ENABLED
    #define YT_USE_SSE42
#endif

#ifdef _win_
    // Someone above has defined this by including one of Windows headers.
    #undef GetMessage
    #undef Yield

    // For protobuf-generated files:
    // C4125: decimal digit terminates octal escape sequence
    #pragma warning (disable: 4125)
    // C4505: unreferenced local function has been removed
    #pragma warning (disable : 4505)
    // C4121: alignment of a member was sensitive to packing
    #pragma warning (disable: 4121)
    // C4503: decorated name length exceeded, name was truncated
    #pragma warning (disable : 4503)
    // C4714: function marked as __forceinline not inlined
    #pragma warning (disable: 4714)
    // C4250: inherits via dominance
    #pragma warning (disable: 4250)
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define YT_ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
    // Prevent GCC from throwing out functions in release builds.
    #define YT_ATTRIBUTE_USED __attribute__((used))
#elif defined(_MSC_VER)
    #define YT_ATTRIBUTE_NO_SANITIZE_ADDRESS
    #define YT_ATTRIBUTE_USED
#else
    #error Unsupported compiler
#endif

#if defined(_unix_)
    #define YT_ATTRIBUTE_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
    #define YT_ATTRIBUTE_NO_UNIQUE_ADDRESS
#endif
