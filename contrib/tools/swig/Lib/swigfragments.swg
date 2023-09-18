/* -----------------------------------------------------------------------------
 * swigfragments.swg
 *
 * Common fragments
 * ----------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * Fragments for C header files
 * ----------------------------------------------------------------------------- */

%fragment("<float.h>", "header") %{
#include <float.h>
%}

/* Default compiler options for gcc allow long_long but not LLONG_MAX. 
 * Define SWIG_NO_LLONG_MAX if this added limits support is not wanted. */
%fragment("<limits.h>", "header") %{
#include <limits.h>
#if !defined(SWIG_NO_LLONG_MAX)
# if !defined(LLONG_MAX) && defined(__GNUC__) && defined (__LONG_LONG_MAX__)
#   define LLONG_MAX __LONG_LONG_MAX__
#   define LLONG_MIN (-LLONG_MAX - 1LL)
#   define ULLONG_MAX (LLONG_MAX * 2ULL + 1ULL)
# endif
#endif
%}

%fragment("<math.h>", "header") %{
#include <math.h>
%}

%fragment("<string.h>", "header") %{
#include <string.h>
%}

%fragment("<stddef.h>", "header") %{
#include <stddef.h>
%}

%fragment("<stdio.h>", "header") %{
#include <stdio.h>
#if (defined(_MSC_VER) && (_MSC_VER < 1900)) || defined(__BORLANDC__) || defined(_WATCOM)
# ifndef snprintf
#  define snprintf _snprintf
# endif
#endif
%}

%fragment("<stdlib.h>", "header") %{
#include <stdlib.h>
#ifdef _MSC_VER
# ifndef strtoull
#  define strtoull _strtoui64
# endif
# ifndef strtoll
#  define strtoll _strtoi64
# endif
#endif
%}

%fragment("<wchar.h>", "header") %{
#include <wchar.h>
#include <limits.h>
#ifndef WCHAR_MIN
#  define WCHAR_MIN 0
#endif
#ifndef WCHAR_MAX
#  define WCHAR_MAX 65535
#endif
%}

/* -----------------------------------------------------------------------------
 * Fragments for C++ header files
 * ----------------------------------------------------------------------------- */

%fragment("<algorithm>", "header") %{
#include <algorithm>
%}

%fragment("<stdexcept>", "header") %{
#include <stdexcept>
%}

%fragment("<string>", "header") %{
#include <string>
%}

%fragment("<memory>", "header") %{
#include <memory>
%}
