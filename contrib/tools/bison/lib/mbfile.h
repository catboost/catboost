/* Multibyte character I/O: macros for multi-byte encodings.
   Copyright (C) 2001, 2005, 2009-2020 Free Software Foundation, Inc.

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

/* Written by Mitsuru Chinen <mchinen@yamato.ibm.com>
   and Bruno Haible <bruno@clisp.org>.  */

/* The macros in this file implement multi-byte character input from a
   stream.

   mb_file_t
     is the type for multibyte character input stream, usable for variable
     declarations.

   mbf_char_t
     is the type for multibyte character or EOF, usable for variable
     declarations.

   mbf_init (mbf, stream)
     initializes the MB_FILE for reading from stream.

   mbf_getc (mbc, mbf)
     reads the next multibyte character from mbf and stores it in mbc.

   mb_iseof (mbc)
     returns true if mbc represents the EOF value.

   Here are the function prototypes of the macros.

   extern void          mbf_init (mb_file_t mbf, FILE *stream);
   extern void          mbf_getc (mbf_char_t mbc, mb_file_t mbf);
   extern bool          mb_iseof (const mbf_char_t mbc);
 */

#ifndef _MBFILE_H
#define _MBFILE_H 1

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/* Tru64 with Desktop Toolkit C has a bug: <stdio.h> must be included before
   <wchar.h>.
   BSD/OS 4.1 has a bug: <stdio.h> and <time.h> must be included before
   <wchar.h>.  */
#include <stdio.h>
#include <time.h>
#include <wchar.h>

#include "mbchar.h"

#ifndef _GL_INLINE_HEADER_BEGIN
 #error "Please include config.h first."
#endif
_GL_INLINE_HEADER_BEGIN
#ifndef MBFILE_INLINE
# define MBFILE_INLINE _GL_INLINE
#endif

struct mbfile_multi {
  FILE *fp;
  bool eof_seen;
  bool have_pushback;
  mbstate_t state;
  unsigned int bufcount;
  char buf[MBCHAR_BUF_SIZE];
  struct mbchar pushback;
};

MBFILE_INLINE void
mbfile_multi_getc (struct mbchar *mbc, struct mbfile_multi *mbf)
{
  size_t bytes;

  /* If EOF has already been seen, don't use getc.  This matters if
     mbf->fp is connected to an interactive tty.  */
  if (mbf->eof_seen)
    goto eof;

  /* Return character pushed back, if there is one.  */
  if (mbf->have_pushback)
    {
      mb_copy (mbc, &mbf->pushback);
      mbf->have_pushback = false;
      return;
    }

  /* Before using mbrtowc, we need at least one byte.  */
  if (mbf->bufcount == 0)
    {
      int c = getc (mbf->fp);
      if (c == EOF)
        {
          mbf->eof_seen = true;
          goto eof;
        }
      mbf->buf[0] = (unsigned char) c;
      mbf->bufcount++;
    }

  /* Handle most ASCII characters quickly, without calling mbrtowc().  */
  if (mbf->bufcount == 1 && mbsinit (&mbf->state) && is_basic (mbf->buf[0]))
    {
      /* These characters are part of the basic character set.  ISO C 99
         guarantees that their wide character code is identical to their
         char code.  */
      mbc->wc = mbc->buf[0] = mbf->buf[0];
      mbc->wc_valid = true;
      mbc->ptr = &mbc->buf[0];
      mbc->bytes = 1;
      mbf->bufcount = 0;
      return;
    }

  /* Use mbrtowc on an increasing number of bytes.  Read only as many bytes
     from mbf->fp as needed.  This is needed to give reasonable interactive
     behaviour when mbf->fp is connected to an interactive tty.  */
  for (;;)
    {
      /* We don't know whether the 'mbrtowc' function updates the state when
         it returns -2, - this is the ISO C 99 and glibc-2.2 behaviour - or
         not - amended ANSI C, glibc-2.1 and Solaris 2.7 behaviour.  We
         don't have an autoconf test for this, yet.
         The new behaviour would allow us to feed the bytes one by one into
         mbrtowc.  But the old behaviour forces us to feed all bytes since
         the end of the last character into mbrtowc.  Since we want to retry
         with more bytes when mbrtowc returns -2, we must backup the state
         before calling mbrtowc, because implementations with the new
         behaviour will clobber it.  */
      mbstate_t backup_state = mbf->state;

      bytes = mbrtowc (&mbc->wc, &mbf->buf[0], mbf->bufcount, &mbf->state);

      if (bytes == (size_t) -1)
        {
          /* An invalid multibyte sequence was encountered.  */
          /* Return a single byte.  */
          bytes = 1;
          mbc->wc_valid = false;
          break;
        }
      else if (bytes == (size_t) -2)
        {
          /* An incomplete multibyte character.  */
          mbf->state = backup_state;
          if (mbf->bufcount == MBCHAR_BUF_SIZE)
            {
              /* An overlong incomplete multibyte sequence was encountered.  */
              /* Return a single byte.  */
              bytes = 1;
              mbc->wc_valid = false;
              break;
            }
          else
            {
              /* Read one more byte and retry mbrtowc.  */
              int c = getc (mbf->fp);
              if (c == EOF)
                {
                  /* An incomplete multibyte character at the end.  */
                  mbf->eof_seen = true;
                  bytes = mbf->bufcount;
                  mbc->wc_valid = false;
                  break;
                }
              mbf->buf[mbf->bufcount] = (unsigned char) c;
              mbf->bufcount++;
            }
        }
      else
        {
          if (bytes == 0)
            {
              /* A null wide character was encountered.  */
              bytes = 1;
              assert (mbf->buf[0] == '\0');
              assert (mbc->wc == 0);
            }
          mbc->wc_valid = true;
          break;
        }
    }

  /* Return the multibyte sequence mbf->buf[0..bytes-1].  */
  mbc->ptr = &mbc->buf[0];
  memcpy (&mbc->buf[0], &mbf->buf[0], bytes);
  mbc->bytes = bytes;

  mbf->bufcount -= bytes;
  if (mbf->bufcount > 0)
    {
      /* It's not worth calling memmove() for so few bytes.  */
      unsigned int count = mbf->bufcount;
      char *p = &mbf->buf[0];

      do
        {
          *p = *(p + bytes);
          p++;
        }
      while (--count > 0);
    }
  return;

eof:
  /* An mbchar_t with bytes == 0 is used to indicate EOF.  */
  mbc->ptr = NULL;
  mbc->bytes = 0;
  mbc->wc_valid = false;
  return;
}

MBFILE_INLINE void
mbfile_multi_ungetc (const struct mbchar *mbc, struct mbfile_multi *mbf)
{
  mb_copy (&mbf->pushback, mbc);
  mbf->have_pushback = true;
}

typedef struct mbfile_multi mb_file_t;

typedef mbchar_t mbf_char_t;

#define mbf_init(mbf, stream)                                           \
  ((mbf).fp = (stream),                                                 \
   (mbf).eof_seen = false,                                              \
   (mbf).have_pushback = false,                                         \
   memset (&(mbf).state, '\0', sizeof (mbstate_t)),                     \
   (mbf).bufcount = 0)

#define mbf_getc(mbc, mbf) mbfile_multi_getc (&(mbc), &(mbf))

#define mbf_ungetc(mbc, mbf) mbfile_multi_ungetc (&(mbc), &(mbf))

#define mb_iseof(mbc) ((mbc).bytes == 0)

_GL_INLINE_HEADER_END

#endif /* _MBFILE_H */
