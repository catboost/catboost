/* Unicode character output to streams with locale dependent encoding.

   Copyright (C) 2000-2003, 2006, 2008-2020 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* Written by Bruno Haible <haible@clisp.cons.org>.  */

#include <config.h>

/* Specification.  */
#include "unicodeio.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>

#if HAVE_ICONV
# include <iconv.h>
#endif

#include <error.h>

#include "gettext.h"
#define _(msgid) gettext (msgid)
#define N_(msgid) msgid

#include "localcharset.h"
#include "unistr.h"

/* When we pass a Unicode character to iconv(), we must pass it in a
   suitable encoding. The standardized Unicode encodings are
   UTF-8, UCS-2, UCS-4, UTF-16, UTF-16BE, UTF-16LE, UTF-7.
   UCS-2 supports only characters up to \U0000FFFF.
   UTF-16 and variants support only characters up to \U0010FFFF.
   UTF-7 is way too complex and not supported by glibc-2.1.
   UCS-4 specification leaves doubts about endianness and byte order
   mark. glibc currently interprets it as big endian without byte order
   mark, but this is not backed by an RFC.
   So we use UTF-8. It supports characters up to \U7FFFFFFF and is
   unambiguously defined.  */

/* Luckily, the encoding's name is platform independent.  */
#define UTF8_NAME "UTF-8"

/* Converts the Unicode character CODE to its multibyte representation
   in the current locale and calls the SUCCESS callback on the resulting
   byte sequence.  If an error occurs, invokes the FAILURE callback instead,
   passing it CODE and an English error string.
   Returns whatever the callback returned.
   Assumes that the locale doesn't change between two calls.  */
long
unicode_to_mb (unsigned int code,
               long (*success) (const char *buf, size_t buflen,
                                void *callback_arg),
               long (*failure) (unsigned int code, const char *msg,
                                void *callback_arg),
               void *callback_arg)
{
  static int initialized;
  static int is_utf8;
#if HAVE_ICONV
  static iconv_t utf8_to_local;
#endif

  char inbuf[6];
  int count;

  if (!initialized)
    {
      const char *charset = locale_charset ();

      is_utf8 = !strcmp (charset, UTF8_NAME);
#if HAVE_ICONV
      if (!is_utf8)
        {
          utf8_to_local = iconv_open (charset, UTF8_NAME);
          if (utf8_to_local == (iconv_t)(-1))
            /* For an unknown encoding, assume ASCII.  */
            utf8_to_local = iconv_open ("ASCII", UTF8_NAME);
        }
#endif
      initialized = 1;
    }

  /* Test whether the utf8_to_local converter is available at all.  */
  if (!is_utf8)
    {
#if HAVE_ICONV
      if (utf8_to_local == (iconv_t)(-1))
        return failure (code, N_("iconv function not usable"), callback_arg);
#else
      return failure (code, N_("iconv function not available"), callback_arg);
#endif
    }

  /* Convert the character to UTF-8.  */
  count = u8_uctomb ((unsigned char *) inbuf, code, sizeof (inbuf));
  if (count < 0)
    return failure (code, N_("character out of range"), callback_arg);

#if HAVE_ICONV
  if (!is_utf8)
    {
      char outbuf[25];
      const char *inptr;
      size_t inbytesleft;
      char *outptr;
      size_t outbytesleft;
      size_t res;

      inptr = inbuf;
      inbytesleft = count;
      outptr = outbuf;
      outbytesleft = sizeof (outbuf);

      /* Convert the character from UTF-8 to the locale's charset.  */
      res = iconv (utf8_to_local,
                   (ICONV_CONST char **)&inptr, &inbytesleft,
                   &outptr, &outbytesleft);
      /* Analyze what iconv() actually did and distinguish replacements
         that are OK (no need to invoke the FAILURE callback), such as
           - replacing GREEK SMALL LETTER MU with MICRO SIGN, or
           - replacing FULLWIDTH COLON with ':', or
           - replacing a Unicode TAG character (U+E00xx) with an empty string,
         from replacements that are worse than the FAILURE callback, such as
           - replacing 'ç' with '?' (NetBSD, Solaris 11) or '*' (musl) or
             NUL (IRIX).  */
      if (inbytesleft > 0 || res == (size_t)(-1)
          /* Irix iconv() inserts a NUL byte if it cannot convert.  */
# if !defined _LIBICONV_VERSION && (defined sgi || defined __sgi)
          || (res > 0 && code != 0 && outptr - outbuf == 1 && *outbuf == '\0')
# endif
          /* NetBSD iconv() and Solaris 11 iconv() insert a '?' if they cannot
             convert.  */
# if !defined _LIBICONV_VERSION && (defined __NetBSD__ || defined __sun)
          || (res > 0 && outptr - outbuf == 1 && *outbuf == '?')
# endif
          /* musl libc iconv() inserts a '*' if it cannot convert.  */
# if !defined _LIBICONV_VERSION && MUSL_LIBC
          || (res > 0 && outptr - outbuf == 1 && *outbuf == '*')
# endif
         )
        return failure (code, NULL, callback_arg);

      /* Avoid glibc-2.1 bug and Solaris 7 bug.  */
# if defined _LIBICONV_VERSION \
    || !(((__GLIBC__ - 0 == 2 && __GLIBC_MINOR__ - 0 <= 1) \
          && !defined __UCLIBC__) \
         || defined __sun)

      /* Get back to the initial shift state.  */
      res = iconv (utf8_to_local, NULL, NULL, &outptr, &outbytesleft);
      if (res == (size_t)(-1))
        return failure (code, NULL, callback_arg);
# endif

      return success (outbuf, outptr - outbuf, callback_arg);
    }
#endif

  /* At this point, is_utf8 is true, so no conversion is needed.  */
  return success (inbuf, count, callback_arg);
}

/* Simple success callback that outputs the converted string.
   The STREAM is passed as callback_arg.  */
long
fwrite_success_callback (const char *buf, size_t buflen, void *callback_arg)
{
  FILE *stream = (FILE *) callback_arg;

  /* The return value of fwrite can be ignored here, because under normal
     conditions (STREAM is an open stream and not wide-character oriented)
     when fwrite() returns a value != buflen it also sets STREAM's error
     indicator.  */
  fwrite (buf, 1, buflen, stream);
  return 0;
}

/* Simple failure callback that displays an error and exits.  */
static long
exit_failure_callback (unsigned int code, const char *msg,
                       void *callback_arg _GL_UNUSED)
{
  if (msg == NULL)
    error (1, 0, _("cannot convert U+%04X to local character set"), code);
  else
    error (1, 0, _("cannot convert U+%04X to local character set: %s"), code,
           gettext (msg));
  return -1;
}

/* Simple failure callback that displays a fallback representation in plain
   ASCII, using the same notation as ISO C99 strings.  */
static long
fallback_failure_callback (unsigned int code,
                           const char *msg _GL_UNUSED,
                           void *callback_arg)
{
  FILE *stream = (FILE *) callback_arg;

  if (code < 0x10000)
    fprintf (stream, "\\u%04X", code);
  else
    fprintf (stream, "\\U%08X", code);
  return -1;
}

/* Outputs the Unicode character CODE to the output stream STREAM.
   Upon failure, exit if exit_on_error is true, otherwise output a fallback
   notation.  */
void
print_unicode_char (FILE *stream, unsigned int code, int exit_on_error)
{
  unicode_to_mb (code, fwrite_success_callback,
                 exit_on_error
                 ? exit_failure_callback
                 : fallback_failure_callback,
                 stream);
}
