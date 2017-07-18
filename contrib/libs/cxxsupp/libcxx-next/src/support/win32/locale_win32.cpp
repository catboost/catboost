// -*- C++ -*-
//===-------------------- support/win32/locale_win32.cpp ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <locale>
#include <cstdarg> // va_start, va_end

using std::__libcpp_locale_guard;

// FIXME: base currently unused. Needs manual work to construct the new locale
locale_t newlocale( int mask, const char * locale, locale_t /*base*/ )
{
    return {_create_locale( LC_ALL, locale ), locale};
}

// uselocale can't be implemented on Windows because Windows allows partial modification
// of thread-local locale and so _get_current_locale() returns a copy while uselocale does
// not create any copies.
// We can still implement raii even without uselocale.

decltype(MB_CUR_MAX) MB_CUR_MAX_L( locale_t __l )
{
#if defined(_LIBCPP_MSVCRT)
  return ___mb_cur_max_l_func(__l);
#else
  __libcpp_locale_guard __curent(__l);
  return MB_CUR_MAX;
#endif
}


lconv *localeconv_l( locale_t loc )
{
    __libcpp_locale_guard __current( loc );
    return localeconv();
}
size_t mbrlen_l( const char *__restrict s, size_t n,
                 mbstate_t *__restrict ps, locale_t loc )
{
    __libcpp_locale_guard __current( loc );
    return mbrlen( s, n, ps );
}
size_t mbsrtowcs_l( wchar_t *__restrict dst, const char **__restrict src,
                    size_t len, mbstate_t *__restrict ps, locale_t loc )
{
    __libcpp_locale_guard __current( loc );
    return mbsrtowcs( dst, src, len, ps );
}
size_t wcrtomb_l( char *__restrict s, wchar_t wc, mbstate_t *__restrict ps,
                  locale_t loc )
{
    __libcpp_locale_guard __current( loc );
    return wcrtomb( s, wc, ps );
}
size_t mbrtowc_l( wchar_t *__restrict pwc, const char *__restrict s,
                  size_t n, mbstate_t *__restrict ps, locale_t loc )
{
    __libcpp_locale_guard __current( loc );
    return mbrtowc( pwc, s, n, ps );
}
size_t mbsnrtowcs_l( wchar_t *__restrict dst, const char **__restrict src,
                     size_t nms, size_t len, mbstate_t *__restrict ps, locale_t loc )
{
    __libcpp_locale_guard __current( loc );
    return mbsnrtowcs( dst, src, nms, len, ps );
}
size_t wcsnrtombs_l( char *__restrict dst, const wchar_t **__restrict src,
                     size_t nwc, size_t len, mbstate_t *__restrict ps, locale_t loc )
{
    __libcpp_locale_guard __current( loc );
    return wcsnrtombs( dst, src, nwc, len, ps );
}
wint_t btowc_l( int c, locale_t loc )
{
    __libcpp_locale_guard __current( loc );
    return btowc( c );
}
int wctob_l( wint_t c, locale_t loc )
{
    __libcpp_locale_guard __current( loc );
    return wctob( c );
}

int snprintf_l(char *ret, size_t n, locale_t loc, const char *format, ...)
{
    __libcpp_locale_guard __current( loc );
    va_list ap;
    va_start( ap, format );
    int result = vsnprintf( ret, n, format, ap );
    va_end(ap);
    return result;
}

int asprintf_l( char **ret, locale_t loc, const char *format, ... )
{
    va_list ap;
    va_start( ap, format );
    int result = vasprintf_l( ret, loc, format, ap );
    va_end(ap);
    return result;
}

// Adaptation of vasprintf from support.cpp
// Like sprintf, but when return value >= 0 it returns
// a pointer to a malloc'd string in *sptr.
// If return >= 0, use free to delete *sptr.
int vasprintf_l( char **sptr, locale_t loc, const char *__restrict format, va_list ap )
{
    *sptr = NULL;
    // Query the count required.
    int count = _vsnprintf_l( NULL, 0, format, loc, ap );
    if (count < 0)
        return count;
    size_t buffer_size = static_cast<size_t>(count) + 1;
    char* p = static_cast<char*>(malloc(buffer_size));
    if ( ! p )
        return -1;
    // If we haven't used exactly what was required, something is wrong.
    // Maybe bug in vsnprintf. Report the error and return.
    if (_vsnprintf_l(p, buffer_size, format, loc, ap) != count) {
        free(p);
        return -1;
    }
    // All good. This is returning memory to the caller not freeing it.
    *sptr = p;
    return count;
}

#if !defined(_LIBCPP_MSVCRT)
float strtof_l(const char* nptr, char** endptr, locale_t loc) {
  __libcpp_locale_guard __current(loc);
  return strtof(nptr, endptr);
}

long double strtold_l(const char* nptr, char** endptr, locale_t loc) {
  __libcpp_locale_guard __current(loc);
  return strtold(nptr, endptr);
}
#endif
