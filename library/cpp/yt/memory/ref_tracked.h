#pragma once

#include <library/cpp/yt/misc/port.h>
#include <library/cpp/yt/misc/source_location.h>
#include <library/cpp/yt/misc/strong_typedef.h>

#include <util/system/defaults.h>

#include <atomic>
#include <typeinfo>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

YT_DEFINE_STRONG_TYPEDEF(TRefCountedTypeCookie, int)
constexpr TRefCountedTypeCookie NullRefCountedTypeCookie{-1};

YT_DEFINE_STRONG_TYPEDEF(TRefCountedTypeKey, const std::type_info*)

////////////////////////////////////////////////////////////////////////////////

// Used to avoid including heavy ref_counted_tracker.h
class TRefCountedTrackerFacade
{
public:
    static TRefCountedTypeCookie GetCookie(
        TRefCountedTypeKey typeKey,
        size_t instanceSize,
        const NYT::TSourceLocation& location);

    static void AllocateInstance(TRefCountedTypeCookie cookie);
    static void FreeInstance(TRefCountedTypeCookie cookie);

    static void AllocateTagInstance(TRefCountedTypeCookie cookie);
    static void FreeTagInstance(TRefCountedTypeCookie cookie);

    static void AllocateSpace(TRefCountedTypeCookie cookie, size_t size);
    static void FreeSpace(TRefCountedTypeCookie cookie, size_t size);

    // Typically invoked from GDB console.
    // Dumps the ref-counted statistics sorted by "bytes alive".
    static void Dump();
};

////////////////////////////////////////////////////////////////////////////////

namespace {

//! A per-translation unit tag type.
struct TCurrentTranslationUnitTag
{ };

} // namespace

template <class T>
TRefCountedTypeKey GetRefCountedTypeKey();

template <class T>
TRefCountedTypeCookie GetRefCountedTypeCookie();

template <class T, class TTag, int Counter>
TRefCountedTypeCookie GetRefCountedTypeCookieWithLocation(
    const TSourceLocation& location);

////////////////////////////////////////////////////////////////////////////////

//! A lightweight mix-in that integrates any class into TRefCountedTracker statistics.
/*!
 *  |T| must be the actual derived type.
 *
 *  This mix-in provides statistical tracking only, |T| is responsible for implementing
 *  lifetime management on its own.
 */
template <class T>
class TRefTracked
{
public:
#ifdef YT_ENABLE_REF_COUNTED_TRACKING
    TRefTracked();
    TRefTracked(const TRefTracked&);
    TRefTracked(TRefTracked&&);
    ~TRefTracked();
#endif
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define REF_TRACKED_INL_H_
#include "ref_tracked-inl.h"
#undef REF_TRACKED_INL_H_
