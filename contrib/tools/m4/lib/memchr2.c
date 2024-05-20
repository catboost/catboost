/* Copyright (C) 1991, 1993, 1996-1997, 1999-2000, 2003-2004, 2006, 2008-2013
   Free Software Foundation, Inc.

   Based on strlen implementation by Torbjorn Granlund (tege@sics.se),
   with help from Dan Sahlin (dan@sics.se) and
   commentary by Jim Blandy (jimb@ai.mit.edu);
   adaptation to memchr suggested by Dick Karpinski (dick@cca.ucsf.edu),
   and implemented in glibc by Roland McGrath (roland@ai.mit.edu).
   Extension to memchr2 implemented by Eric Blake (ebb9@byu.net).

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3 of the License, or any
later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include <config.h>

#include "memchr2.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

/* Return the first address of either C1 or C2 (treated as unsigned
   char) that occurs within N bytes of the memory region S.  If
   neither byte appears, return NULL.  */
void *
memchr2 (void const *s, int c1_in, int c2_in, size_t n)
{
  /* On 32-bit hardware, choosing longword to be a 32-bit unsigned
     long instead of a 64-bit uintmax_t tends to give better
     performance.  On 64-bit hardware, unsigned long is generally 64
     bits already.  Change this typedef to experiment with
     performance.  */
  typedef unsigned long int longword;

  const unsigned char *char_ptr;
  void const *void_ptr;
  const longword *longword_ptr;
  longword repeated_one;
  longword repeated_c1;
  longword repeated_c2;
  unsigned char c1;
  unsigned char c2;

  c1 = (unsigned char) c1_in;
  c2 = (unsigned char) c2_in;

  if (c1 == c2)
    return memchr (s, c1, n);

  /* Handle the first few bytes by reading one byte at a time.
     Do this until VOID_PTR is aligned on a longword boundary.  */
  for (void_ptr = s;
       n > 0 && (uintptr_t) void_ptr % sizeof (longword) != 0;
       --n)
    {
      char_ptr = void_ptr;
      if (*char_ptr == c1 || *char_ptr == c2)
        return (void *) void_ptr;
      void_ptr = char_ptr + 1;
    }

  longword_ptr = void_ptr;

  /* All these elucidatory comments refer to 4-byte longwords,
     but the theory applies equally well to any size longwords.  */

  /* Compute auxiliary longword values:
     repeated_one is a value which has a 1 in every byte.
     repeated_c1 has c1 in every byte.
     repeated_c2 has c2 in every byte.  */
  repeated_one = 0x01010101;
  repeated_c1 = c1 | (c1 << 8);
  repeated_c2 = c2 | (c2 << 8);
  repeated_c1 |= repeated_c1 << 16;
  repeated_c2 |= repeated_c2 << 16;
  if (0xffffffffU < (longword) -1)
    {
      repeated_one |= repeated_one << 31 << 1;
      repeated_c1 |= repeated_c1 << 31 << 1;
      repeated_c2 |= repeated_c2 << 31 << 1;
      if (8 < sizeof (longword))
        {
          size_t i;

          for (i = 64; i < sizeof (longword) * 8; i *= 2)
            {
              repeated_one |= repeated_one << i;
              repeated_c1 |= repeated_c1 << i;
              repeated_c2 |= repeated_c2 << i;
            }
        }
    }

  /* Instead of the traditional loop which tests each byte, we will test a
     longword at a time.  The tricky part is testing if *any of the four*
     bytes in the longword in question are equal to c1 or c2.  We first use
     an xor with repeated_c1 and repeated_c2, respectively.  This reduces
     the task to testing whether *any of the four* bytes in longword1 or
     longword2 is zero.

     Let's consider longword1.  We compute tmp1 =
       ((longword1 - repeated_one) & ~longword1) & (repeated_one << 7).
     That is, we perform the following operations:
       1. Subtract repeated_one.
       2. & ~longword1.
       3. & a mask consisting of 0x80 in every byte.
     Consider what happens in each byte:
       - If a byte of longword1 is zero, step 1 and 2 transform it into 0xff,
         and step 3 transforms it into 0x80.  A carry can also be propagated
         to more significant bytes.
       - If a byte of longword1 is nonzero, let its lowest 1 bit be at
         position k (0 <= k <= 7); so the lowest k bits are 0.  After step 1,
         the byte ends in a single bit of value 0 and k bits of value 1.
         After step 2, the result is just k bits of value 1: 2^k - 1.  After
         step 3, the result is 0.  And no carry is produced.
     So, if longword1 has only non-zero bytes, tmp1 is zero.
     Whereas if longword1 has a zero byte, call j the position of the least
     significant zero byte.  Then the result has a zero at positions 0, ...,
     j-1 and a 0x80 at position j.  We cannot predict the result at the more
     significant bytes (positions j+1..3), but it does not matter since we
     already have a non-zero bit at position 8*j+7.

     Similarly, we compute tmp2 =
       ((longword2 - repeated_one) & ~longword2) & (repeated_one << 7).

     The test whether any byte in longword1 or longword2 is zero is equivalent
     to testing whether tmp1 is nonzero or tmp2 is nonzero.  We can combine
     this into a single test, whether (tmp1 | tmp2) is nonzero.  */

  while (n >= sizeof (longword))
    {
      longword longword1 = *longword_ptr ^ repeated_c1;
      longword longword2 = *longword_ptr ^ repeated_c2;

      if (((((longword1 - repeated_one) & ~longword1)
            | ((longword2 - repeated_one) & ~longword2))
           & (repeated_one << 7)) != 0)
        break;
      longword_ptr++;
      n -= sizeof (longword);
    }

  char_ptr = (const unsigned char *) longword_ptr;

  /* At this point, we know that either n < sizeof (longword), or one of the
     sizeof (longword) bytes starting at char_ptr is == c1 or == c2.  On
     little-endian machines, we could determine the first such byte without
     any further memory accesses, just by looking at the (tmp1 | tmp2) result
     from the last loop iteration.  But this does not work on big-endian
     machines.  Choose code that works in both cases.  */

  for (; n > 0; --n, ++char_ptr)
    {
      if (*char_ptr == c1 || *char_ptr == c2)
        return (void *) char_ptr;
    }

  return NULL;
}
