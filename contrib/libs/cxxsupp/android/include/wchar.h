/*
  Copyright (C) 2005-2012 Rich Felker

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

  Modified in 2013 for the Android Open Source Project.
 */
#ifndef NDK_ANDROID_SUPPORT_WCHAR_H
#define NDK_ANDROID_SUPPORT_WCHAR_H

/* IMPORTANT NOTE: Unlike other headers in the support library, this
 * one doesn't try to include the Bionic header through #include_next.
 *
 * This is intentional, and comes from the fact that before Gingerbread,
 * i.e. API level 9, the platform didn't really support wide chars, more
 * precisely:
 *    - wchar_t is defined as an 8-bit unsigned integer.
 *    - the few wchar-related functions available are just stubs
 *      to their 8-bit counterparts (e.g. wcslen() -> strlen()).
 *
 * Starting from API level 9, wchar_t is a 32-bit unsigned integer,
 * and wchar-related functions implement support for it with several
 * gotchas:
 *    - no proper Unicode support (e.g. towlower() only works on ASCII
 *      codepoints, ignores all others).
 *
 *    - no wprintf() and wscanf() functionality.
 *
 *    - no multi-byte conversion routines.
 *
 * By completely overriding the C library functions, the support library
 * can be used to generate code that will run properly on _any_ version
 * of Android.
 *
 * This implementation supports the following:
 *
 *    - Unicode code points in wchar_t, and working towlower() / towupper()
 *      using the en_US.UTF-8 case mappings.
 *
 *    - Multi-byte encoding/decoding to/from UTF-8 (no other multibyte
 *      encoding are supported).
 *
 *    - wprintf() / wfprintf() support.
 *
 *    - wscanf() / wfscanf() coming soon :)
 */
#if defined(__LP64__)

#include_next <wchar.h>

#else

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>  // for va_list
#include <stdio.h>   // for FILE
#include <stddef.h>  // for size_t
#include <wctype.h>
#include <xlocale.h> // for locale_t

#define __need___wchar_t
#include <stddef.h>

#ifndef WCHAR_MAX
#define WCHAR_MAX __WCHAR_MAX__
/* Clang does not define __WCHAR_MIN__ */
#if defined(__WCHAR_UNSIGNED__)
#define WCHAR_MIN L'\0'
#else
#define WCHAR_MIN (-(WCHAR_MAX) - 1)
#endif
#endif

#define WEOF ((wint_t)(-1))

typedef struct
{
  unsigned __opaque1, __opaque2;
} mbstate_t;

wchar_t *wcscpy (wchar_t *__restrict__, const wchar_t *__restrict__);
wchar_t *wcsncpy (wchar_t *__restrict__, const wchar_t *__restrict__, size_t);

wchar_t *wcscat (wchar_t *__restrict__, const wchar_t *__restrict__);
wchar_t *wcsncat (wchar_t *__restrict__, const wchar_t *__restrict__, size_t);

int wcscmp (const wchar_t *, const wchar_t *);
int wcsncmp (const wchar_t *, const wchar_t *, size_t);

int wcscoll(const wchar_t *, const wchar_t *);
size_t wcsxfrm (wchar_t *__restrict__, const wchar_t *__restrict__, size_t n);

wchar_t *wcschr (const wchar_t *, wchar_t);
wchar_t *wcsrchr (const wchar_t *, wchar_t);

size_t wcscspn (const wchar_t *, const wchar_t *);
size_t wcsspn (const wchar_t *, const wchar_t *);
wchar_t *wcspbrk (const wchar_t *, const wchar_t *);

wchar_t *wcstok (wchar_t *__restrict__, const wchar_t *__restrict__, wchar_t **__restrict__);

size_t wcslen (const wchar_t *);

wchar_t *wcsstr (const wchar_t *__restrict__, const wchar_t *__restrict__);
wchar_t *wcswcs (const wchar_t *, const wchar_t *);

wchar_t *wmemchr (const wchar_t *, wchar_t, size_t);
int wmemcmp (const wchar_t *, const wchar_t *, size_t);
wchar_t *wmemcpy (wchar_t *__restrict__, const wchar_t *__restrict__, size_t);
wchar_t *wmemmove (wchar_t *, const wchar_t *, size_t);
wchar_t *wmemset (wchar_t *, wchar_t, size_t);

wint_t btowc (int);
int wctob (wint_t);

int mbsinit (const mbstate_t *);
size_t mbrtowc (wchar_t *__restrict__, const char *__restrict__, size_t, mbstate_t *__restrict__);
size_t wcrtomb (char *__restrict__, wchar_t, mbstate_t *__restrict__);

size_t mbrlen (const char *__restrict__, size_t, mbstate_t *__restrict__);

size_t mbsrtowcs (wchar_t *__restrict__, const char **__restrict__, size_t, mbstate_t *__restrict__);
size_t wcsrtombs (char *__restrict__, const wchar_t **__restrict__, size_t, mbstate_t *__restrict__);

float wcstof (const wchar_t *__restrict__, wchar_t **__restrict__);
double wcstod (const wchar_t *__restrict__, wchar_t **__restrict__);
long double wcstold (const wchar_t *__restrict__, wchar_t **__restrict__);

long wcstol (const wchar_t *__restrict__, wchar_t **__restrict__, int);
unsigned long wcstoul (const wchar_t *__restrict__, wchar_t **__restrict__, int);

long long wcstoll (const wchar_t *__restrict__, wchar_t **__restrict__, int);
unsigned long long wcstoull (const wchar_t *__restrict__, wchar_t **__restrict__, int);
intmax_t wcstoimax (const wchar_t * nptr, wchar_t** endptr , int base);
uintmax_t wcstoumax (const wchar_t * nptr, wchar_t** endptr , int base);


int fwide (FILE *, int);


int wprintf (const wchar_t *__restrict__, ...);
int fwprintf (FILE *__restrict__, const wchar_t *__restrict__, ...);
int swprintf (wchar_t *__restrict__, size_t, const wchar_t *__restrict__, ...);

int vwprintf (const wchar_t *__restrict__, va_list);
int vfwprintf (FILE *__restrict__, const wchar_t *__restrict__, va_list);
int vswprintf (wchar_t *__restrict__, size_t, const wchar_t *__restrict__, va_list);

int wscanf (const wchar_t *__restrict__, ...);
int fwscanf (FILE *__restrict__, const wchar_t *__restrict__, ...);
int swscanf (const wchar_t *__restrict__, const wchar_t *__restrict__, ...);

int vwscanf (const wchar_t *__restrict__, va_list);
int vfwscanf (FILE *__restrict__, const wchar_t *__restrict__, va_list);
int vswscanf (const wchar_t *__restrict__, const wchar_t *__restrict__, va_list);

wint_t fgetwc (FILE *);
wint_t getwc (FILE *);
wint_t getwchar (void);

wint_t fputwc (wchar_t, FILE *);
wint_t putwc (wchar_t, FILE *);
wint_t putwchar (wchar_t);

wchar_t *fgetws (wchar_t *__restrict__, int, FILE *__restrict__);
int fputws (const wchar_t *__restrict__, FILE *__restrict__);

wint_t ungetwc (wint_t, FILE *);

struct tm;
size_t wcsftime (wchar_t *__restrict__, size_t, const wchar_t *__restrict__, const struct tm *__restrict__);

FILE *open_wmemstream(wchar_t **, size_t *);
size_t mbsnrtowcs(wchar_t *__restrict__, const char **__restrict__, size_t, size_t, mbstate_t *__restrict__);
size_t wcsnrtombs(char *__restrict__, const wchar_t **__restrict__, size_t, size_t, mbstate_t *__restrict__);
wchar_t *wcsdup(const wchar_t *);
size_t wcsnlen (const wchar_t *, size_t);
wchar_t *wcpcpy (wchar_t *__restrict__, const wchar_t *__restrict__);
wchar_t *wcpncpy (wchar_t *__restrict__, const wchar_t *__restrict__, size_t);
int wcscasecmp(const wchar_t *, const wchar_t *);
int wcscasecmp_l(const wchar_t *, const wchar_t *, locale_t);
int wcsncasecmp(const wchar_t *, const wchar_t *, size_t);
int wcsncasecmp_l(const wchar_t *, const wchar_t *, size_t, locale_t);
int wcwidth (wchar_t);
int wcswidth (const wchar_t *, size_t);
int       iswalnum(wint_t);
int       iswalpha(wint_t);
int       iswblank(wint_t);
int       iswcntrl(wint_t);
int       iswdigit(wint_t);
int       iswgraph(wint_t);
int       iswlower(wint_t);
int       iswprint(wint_t);
int       iswpunct(wint_t);
int       iswspace(wint_t);
int       iswupper(wint_t);
int       iswxdigit(wint_t);
int       iswctype(wint_t, wctype_t);
wint_t    towlower(wint_t);
wint_t    towupper(wint_t);
wctype_t  wctype(const char *);

int wcscoll_l(const wchar_t *, const wchar_t *, locale_t);
size_t wcsxfrm_l(wchar_t *__restrict__, const wchar_t *__restrict__, size_t n, locale_t);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // !__LP64__

#endif  // NDK_ANDROID_SUPPORT_WCHAR_H
