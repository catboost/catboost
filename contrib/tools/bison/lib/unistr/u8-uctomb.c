/* Store a character in UTF-8 string.
   Copyright (C) 2002, 2005-2006, 2009-2020 Free Software Foundation, Inc.
   Written by Bruno Haible <bruno@clisp.org>, 2002.

   This program is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published
   by the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#include <config.h>

#if defined IN_LIBUNISTRING
/* Tell unistr.h to declare u8_uctomb as 'extern', not 'static inline'.  */
# error #include "unistring-notinline.h"
#endif

/* Specification.  */
#include "unistr.h"

#include "attribute.h"

#if !HAVE_INLINE

int
u8_uctomb (uint8_t *s, ucs4_t uc, ptrdiff_t n)
{
  if (uc < 0x80)
    {
      if (n > 0)
        {
          s[0] = uc;
          return 1;
        }
      /* else return -2, below.  */
    }
  else
    {
      int count;

      if (uc < 0x800)
        count = 2;
      else if (uc < 0x10000)
        {
          if (uc < 0xd800 || uc >= 0xe000)
            count = 3;
          else
            return -1;
        }
      else if (uc < 0x110000)
        count = 4;
      else
        return -1;

      if (n >= count)
        {
          switch (count) /* note: code falls through cases! */
            {
            case 4: s[3] = 0x80 | (uc & 0x3f); uc = uc >> 6; uc |= 0x10000;
              FALLTHROUGH;
            case 3: s[2] = 0x80 | (uc & 0x3f); uc = uc >> 6; uc |= 0x800;
              FALLTHROUGH;
            case 2: s[1] = 0x80 | (uc & 0x3f); uc = uc >> 6; uc |= 0xc0;
          /*case 1:*/ s[0] = uc;
            }
          return count;
        }
    }
  return -2;
}

#endif
