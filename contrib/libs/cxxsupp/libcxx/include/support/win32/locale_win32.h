// -*- C++ -*-
//===--------------------- support/win32/locale_win32.h -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_WIN32_LOCALE_WIN32_H
#define _LIBCPP_SUPPORT_WIN32_LOCALE_WIN32_H

#include <__config>
#include <stdio.h>
#include <xlocinfo.h> // _locale_t
#include <__nullptr>

#define LC_COLLATE_MASK _M_COLLATE
#define LC_CTYPE_MASK _M_CTYPE
#define LC_MONETARY_MASK _M_MONETARY
#define LC_NUMERIC_MASK _M_NUMERIC
#define LC_TIME_MASK _M_TIME
#define LC_MESSAGES_MASK _M_MESSAGES
#define LC_MESSAGES _X_MESSAGES
#define LC_ALL_MASK _M_ALL

class locale_t {
public:
    locale_t()
        : __locale(nullptr), __locale_str(nullptr) {}
    locale_t(std::nullptr_t)
        : __locale(nullptr), __locale_str(nullptr) {}
    locale_t(_locale_t __xlocale, const char* __xlocale_str)
        : __locale(__xlocale), __locale_str(__xlocale_str) {}

    friend bool operator==(const locale_t& __left, const locale_t& __right) {
        return __left.__locale == __right.__locale;
    }

    friend bool operator==(const locale_t& __left, int __right) {
        return __left.__locale == nullptr && __right == 0;
    }

    friend bool operator==(const locale_t& __left, long long __right) {
        return __left.__locale == nullptr && __right == 0;
    }

    friend bool operator==(const locale_t& __left, std::nullptr_t) {
        return __left.__locale == nullptr;
    }

    friend bool operator==(int __left, const locale_t& __right) {
        return __left == 0 && nullptr == __right.__locale;
    }

    friend bool operator==(std::nullptr_t, const locale_t& __right) {
        return nullptr == __right.__locale;
    }

    friend bool operator!=(const locale_t& __left, const locale_t& __right) {
        return !(__left == __right);
    }

    friend bool operator!=(const locale_t& __left, int __right) {
        return !(__left == __right);
    }

    friend bool operator!=(const locale_t& __left, long long __right) {
        return !(__left == __right);
    }

    friend bool operator!=(const locale_t& __left, std::nullptr_t __right) {
        return !(__left == __right);
    }

    friend bool operator!=(int __left, const locale_t& __right) {
        return !(__left == __right);
    }

    friend bool operator!=(std::nullptr_t __left, const locale_t& __right) {
        return !(__left == __right);
    }

    operator bool() const {
        return __locale != nullptr;
    }

    const char* __get_locale() const { return __locale_str; }

    operator _locale_t() const {
        return __locale;
    }
private:
    _locale_t __locale;
    const char* __locale_str;
};

// Locale management functions
#define freelocale _free_locale
// FIXME: base currently unused. Needs manual work to construct the new locale
locale_t newlocale( int mask, const char * locale, locale_t base );
// uselocale can't be implemented on Windows because Windows allows partial modification
// of thread-local locale and so _get_current_locale() returns a copy while uselocale does
// not create any copies.
// We can still implement raii even without uselocale though.


lconv *localeconv_l( locale_t __loc );
size_t mbrlen_l( const char *__restrict __s, size_t __n,
                 mbstate_t *__restrict __ps, locale_t __loc);
size_t mbsrtowcs_l( wchar_t *__restrict __dst, const char **__restrict __src,
                    size_t __len, mbstate_t *__restrict __ps, locale_t __loc );
size_t wcrtomb_l( char *__restrict s, wchar_t __wc, mbstate_t *__restrict __ps,
                  locale_t __loc);
size_t mbrtowc_l( wchar_t *__restrict __pwc, const char *__restrict __s,
                  size_t __n, mbstate_t *__restrict __ps, locale_t __loc);
size_t mbsnrtowcs_l( wchar_t *__restrict __dst, const char **__restrict __src,
                     size_t __nms, size_t __len, mbstate_t *__restrict __ps, locale_t __loc);
size_t wcsnrtombs_l( char *__restrict __dst, const wchar_t **__restrict __src,
                     size_t __nwc, size_t __len, mbstate_t *__restrict __ps, locale_t __loc);
wint_t btowc_l( int __c, locale_t __loc );
int wctob_l( wint_t __c, locale_t __loc );

decltype(MB_CUR_MAX) MB_CUR_MAX_L( locale_t __l );

// the *_l functions are prefixed on Windows, only available for msvcr80+, VS2005+
#define mbtowc_l _mbtowc_l
#define strtoll_l _strtoi64_l
#define strtoull_l _strtoui64_l
#define strtod_l _strtod_l
#if defined(_LIBCPP_MSVCRT)
#define strtof_l _strtof_l
#define strtold_l _strtod_l
#else
float strtof_l(const char*, char**, locale_t);
long double strtold_l(const char*, char**, locale_t);
#endif
#define isupper_l _isupper_l
#define islower_l _islower_l
#define isblank_l _isblank_l
#define isdigit_l _isdigit_l
#define isxdigit_l _isxdigit_l
#define strcoll_l _strcoll_l
#define strxfrm_l _strxfrm_l
#define wcscoll_l _wcscoll_l
#define wcsxfrm_l _wcsxfrm_l
#define toupper_l _toupper_l
#define tolower_l _tolower_l
#define iswspace_l _iswspace_l
#define iswprint_l _iswprint_l
#define iswcntrl_l _iswcntrl_l
#define iswupper_l _iswupper_l
#define iswlower_l _iswlower_l
#define iswblank_l _iswblank_l
#define iswalpha_l _iswalpha_l
#define iswdigit_l _iswdigit_l
#define iswpunct_l _iswpunct_l
#define iswxdigit_l _iswxdigit_l
#define towupper_l _towupper_l
#define towlower_l _towlower_l
#if defined(__MINGW32__) && __MSVCRT_VERSION__ < 0x0800
#define strftime_l( __s, __l, __f, __tm, __loc) strftime( __s, __l, __f, __tm )
#else
#define strftime_l _strftime_l
#endif
#define sscanf_l( __s, __l, __f, ...) _sscanf_l( __s, __f, __l, ##__VA_ARGS__ )
#define sprintf_l( __s, __l, __f, ... ) _sprintf_l( __s, __f, __l, ##__VA_ARGS__ )
#define vsprintf_l( __s, __l, __f, ... ) _vsprintf_l( __s, __f, __l, ##__VA_ARGS__ )
#define vsnprintf_l( __s, __n, __l, __f, ... ) _vsnprintf_l( __s, __n, __f, __l, ##__VA_ARGS__ )
#define snprintf_l(__s, __n, __l, __f, ...) _snprintf_l( __s, __n, __f, __l, ##__VA_ARGS__ )
_LIBCPP_FUNC_VIS int asprintf_l( char **__ret, locale_t __loc, const char *__format, ... );
_LIBCPP_FUNC_VIS int vasprintf_l( char **__ret, locale_t __loc, const char *__format, va_list __ap );

#endif // _LIBCPP_SUPPORT_WIN32_LOCALE_WIN32_H
