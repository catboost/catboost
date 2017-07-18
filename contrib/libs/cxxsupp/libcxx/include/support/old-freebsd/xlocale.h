//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//==----------------------------------------------------------------------===//
//ATTENTION!
//This file is modified copy of support/newlib/xlocale.h
//Changes are only in defines:
// _LIBCPP_SUPPORT_NEWLIB_XLOCALE_H -> _LIBCPP_SUPPORT__LIBCPP_FREEBSD_90_XLOCALE_H
// _NEWLIB_VERSION -> __LIBCPP_FREEBSD_90
//===----------------------------------------------------------------------===//


#ifndef _LIBCPP_SUPPORT__LIBCPP_FREEBSD_90_XLOCALE_H
#define _LIBCPP_SUPPORT__LIBCPP_FREEBSD_90_XLOCALE_H

#if defined(__LIBCPP_FREEBSD_90)

#include <cstdlib>
#include <clocale>
#include <cwctype>
#include <ctype.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *locale_t;
static inline locale_t duplocale(locale_t) {
  return NULL;
}

static inline void freelocale(locale_t) {
}

static inline locale_t newlocale(int, const char *, locale_t) {
  return NULL;
}

static inline locale_t uselocale(locale_t) {
  return NULL;
}

#define LC_COLLATE_MASK  (1 << LC_COLLATE)
#define LC_CTYPE_MASK    (1 << LC_CTYPE)
#define LC_MESSAGES_MASK (1 << LC_MESSAGES)
#define LC_MONETARY_MASK (1 << LC_MONETARY)
#define LC_NUMERIC_MASK  (1 << LC_NUMERIC)
#define LC_TIME_MASK     (1 << LC_TIME)
#define LC_ALL_MASK (LC_COLLATE_MASK|\
                     LC_CTYPE_MASK|\
                     LC_MONETARY_MASK|\
                     LC_NUMERIC_MASK|\
                     LC_TIME_MASK|\
                     LC_MESSAGES_MASK)

// Share implementation with Android's Bionic
#include "../xlocale/xlocale.h"

locale_t __cloc(void);
#ifdef __cplusplus
} // extern "C"
#endif

#endif // __LIBCPP_FREEBSD_90

#endif
