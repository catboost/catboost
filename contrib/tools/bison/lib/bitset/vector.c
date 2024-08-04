/* Variable array bitsets.

   Copyright (C) 2002-2006, 2009-2015, 2018-2020 Free Software Foundation, Inc.

   Contributed by Michael Hayes (m.hayes@elec.canterbury.ac.nz).

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#include <config.h>

#include "bitset/vector.h"

#include <stdlib.h>
#include <string.h>

#include "xalloc.h"

/* This file implements variable size bitsets stored as a variable
   length array of words.  Any unused bits in the last word must be
   zero.

   Note that binary or ternary operations assume that each bitset operand
   has the same size.
*/

static void vbitset_unused_clear (bitset);

static void vbitset_set (bitset, bitset_bindex);
static void vbitset_reset (bitset, bitset_bindex);
static bool vbitset_test (bitset, bitset_bindex);
static bitset_bindex vbitset_list (bitset, bitset_bindex *,
                                   bitset_bindex, bitset_bindex *);
static bitset_bindex vbitset_list_reverse (bitset, bitset_bindex *,
                                           bitset_bindex, bitset_bindex *);

#define VBITSET_N_WORDS(N) (((N) + BITSET_WORD_BITS - 1) / BITSET_WORD_BITS)
#define VBITSET_WORDS(X) ((X)->b.cdata)
#define VBITSET_SIZE(X) ((X)->b.csize)
#define VBITSET_ASIZE(X) ((X)->v.size)

#undef min
#undef max
#define min(a, b) ((a) > (b) ? (b) : (a))
#define max(a, b) ((a) > (b) ? (a) : (b))

static bitset_bindex
vbitset_resize (bitset src, bitset_bindex n_bits)
{
  if (n_bits == BITSET_NBITS_ (src))
    return n_bits;

  bitset_windex oldsize = VBITSET_SIZE (src);
  bitset_windex newsize = VBITSET_N_WORDS (n_bits);

  if (oldsize < newsize)
    {
      /* The bitset needs to grow.  If we already have enough memory
         allocated, then just zero what we need.  */
      if (newsize > VBITSET_ASIZE (src))
        {
          /* We need to allocate more memory.  When oldsize is
             non-zero this means that we are changing the size, so
             grow the bitset 25% larger than requested to reduce
             number of reallocations.  */

          bitset_windex size = oldsize == 0 ? newsize : newsize + newsize / 4;
          VBITSET_WORDS (src)
            = xrealloc (VBITSET_WORDS (src), size * sizeof (bitset_word));
          VBITSET_ASIZE (src) = size;
        }

      memset (VBITSET_WORDS (src) + oldsize, 0,
              (newsize - oldsize) * sizeof (bitset_word));
    }
  else
    {
      /* The bitset needs to shrink.  There's no point deallocating
         the memory unless it is shrinking by a reasonable amount.  */
      if ((oldsize - newsize) >= oldsize / 2)
        {
          void *p
            = realloc (VBITSET_WORDS (src), newsize * sizeof (bitset_word));
          if (p)
            {
              VBITSET_WORDS (src) = p;
              VBITSET_ASIZE (src) = newsize;
            }
        }

      /* Need to prune any excess bits.  FIXME.  */
    }

  VBITSET_SIZE (src) = newsize;
  BITSET_NBITS_ (src) = n_bits;
  return n_bits;
}


/* Set bit BITNO in bitset DST.  */
static void
vbitset_set (bitset dst, bitset_bindex bitno)
{
  bitset_windex windex = bitno / BITSET_WORD_BITS;

  /* Perhaps we should abort.  The user should explicitly call
     bitset_resize since this will not catch the case when we set a
     bit larger than the current size but smaller than the allocated
     size.  */
  vbitset_resize (dst, bitno);

  dst->b.cdata[windex - dst->b.cindex] |=
    (bitset_word) 1 << (bitno % BITSET_WORD_BITS);
}


/* Reset bit BITNO in bitset DST.  */
static void
vbitset_reset (bitset dst ATTRIBUTE_UNUSED, bitset_bindex bitno ATTRIBUTE_UNUSED)
{
  /* We must be accessing outside the cache so the bit is
     zero anyway.  */
}


/* Test bit BITNO in bitset SRC.  */
static bool
vbitset_test (bitset src ATTRIBUTE_UNUSED,
              bitset_bindex bitno ATTRIBUTE_UNUSED)
{
  /* We must be accessing outside the cache so the bit is
     zero anyway.  */
  return false;
}


/* Find list of up to NUM bits set in BSET in reverse order, starting
   from and including NEXT and store in array LIST.  Return with
   actual number of bits found and with *NEXT indicating where search
   stopped.  */
static bitset_bindex
vbitset_list_reverse (bitset src, bitset_bindex *list,
                      bitset_bindex num, bitset_bindex *next)
{
  bitset_word *srcp = VBITSET_WORDS (src);
  bitset_bindex n_bits = BITSET_SIZE_ (src);

  bitset_bindex rbitno = *next;

  /* If num is 1, we could speed things up with a binary search
     of the word of interest.  */

  if (rbitno >= n_bits)
    return 0;

  bitset_bindex count = 0;

  bitset_bindex bitno = n_bits - (rbitno + 1);

  bitset_windex windex = bitno / BITSET_WORD_BITS;
  unsigned bitcnt = bitno % BITSET_WORD_BITS;
  bitset_bindex bitoff = windex * BITSET_WORD_BITS;

  do
    {
      bitset_word word = srcp[windex] << (BITSET_WORD_BITS - 1 - bitcnt);
      for (; word; bitcnt--)
        {
          if (word & BITSET_MSB)
            {
              list[count++] = bitoff + bitcnt;
              if (count >= num)
                {
                  *next = n_bits - (bitoff + bitcnt);
                  return count;
                }
            }
          word <<= 1;
        }
      bitoff -= BITSET_WORD_BITS;
      bitcnt = BITSET_WORD_BITS - 1;
    }
  while (windex--);

  *next = n_bits - (bitoff + 1);
  return count;
}


/* Find list of up to NUM bits set in BSET starting from and including
   *NEXT and store in array LIST.  Return with actual number of bits
   found and with *NEXT indicating where search stopped.  */
static bitset_bindex
vbitset_list (bitset src, bitset_bindex *list,
              bitset_bindex num, bitset_bindex *next)
{
  bitset_windex size = VBITSET_SIZE (src);
  bitset_word *srcp = VBITSET_WORDS (src);
  bitset_bindex bitno = *next;
  bitset_bindex count = 0;

  bitset_windex windex;
  bitset_bindex bitoff;
  bitset_word word;

  if (!bitno)
    {
      /* Many bitsets are zero, so make this common case fast.  */
      for (windex = 0; windex < size && !srcp[windex]; windex++)
        continue;
      if (windex >= size)
        return 0;

      /* If num is 1, we could speed things up with a binary search
         of the current word.  */

      bitoff = windex * BITSET_WORD_BITS;
    }
  else
    {
      if (bitno >= BITSET_SIZE_ (src))
        return 0;

      windex = bitno / BITSET_WORD_BITS;
      bitno = bitno % BITSET_WORD_BITS;

      if (bitno)
        {
          /* Handle the case where we start within a word.
             Most often, this is executed with large bitsets
             with many set bits where we filled the array
             on the previous call to this function.  */

          bitoff = windex * BITSET_WORD_BITS;
          word = srcp[windex] >> bitno;
          for (bitno = bitoff + bitno; word; bitno++)
            {
              if (word & 1)
                {
                  list[count++] = bitno;
                  if (count >= num)
                    {
                      *next = bitno + 1;
                      return count;
                    }
                }
              word >>= 1;
            }
          windex++;
        }
      bitoff = windex * BITSET_WORD_BITS;
    }

  for (; windex < size; windex++, bitoff += BITSET_WORD_BITS)
    {
      if (!(word = srcp[windex]))
        continue;

      if ((count + BITSET_WORD_BITS) < num)
        {
          for (bitno = bitoff; word; bitno++)
            {
              if (word & 1)
                list[count++] = bitno;
              word >>= 1;
            }
        }
      else
        {
          for (bitno = bitoff; word; bitno++)
            {
              if (word & 1)
                {
                  list[count++] = bitno;
                  if (count >= num)
                    {
                      *next = bitno + 1;
                      return count;
                    }
                }
              word >>= 1;
            }
        }
    }

  *next = bitoff;
  return count;
}


/* Ensure that any unused bits within the last word are clear.  */
static inline void
vbitset_unused_clear (bitset dst)
{
  unsigned last_bit = BITSET_SIZE_ (dst) % BITSET_WORD_BITS;
  if (last_bit)
    VBITSET_WORDS (dst)[VBITSET_SIZE (dst) - 1] &=
      ((bitset_word) 1 << last_bit) - 1;
}


static void
vbitset_ones (bitset dst)
{
  bitset_word *dstp = VBITSET_WORDS (dst);
  unsigned bytes = sizeof (bitset_word) * VBITSET_SIZE (dst);

  memset (dstp, -1, bytes);
  vbitset_unused_clear (dst);
}


static void
vbitset_zero (bitset dst)
{
  bitset_word *dstp = VBITSET_WORDS (dst);
  unsigned bytes = sizeof (bitset_word) * VBITSET_SIZE (dst);

  memset (dstp, 0, bytes);
}


static bool
vbitset_empty_p (bitset dst)
{
  bitset_word *dstp = VBITSET_WORDS (dst);

  for (unsigned i = 0; i < VBITSET_SIZE (dst); i++)
    if (dstp[i])
      return false;
  return true;
}


static void
vbitset_copy1 (bitset dst, bitset src)
{
  if (src == dst)
    return;

  vbitset_resize (dst, BITSET_SIZE_ (src));

  bitset_word *srcp = VBITSET_WORDS (src);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_windex ssize = VBITSET_SIZE (src);
  bitset_windex dsize = VBITSET_SIZE (dst);

  memcpy (dstp, srcp, sizeof (bitset_word) * ssize);

  memset (dstp + sizeof (bitset_word) * ssize, 0,
          sizeof (bitset_word) * (dsize - ssize));
}


static void
vbitset_not (bitset dst, bitset src)
{
  vbitset_resize (dst, BITSET_SIZE_ (src));

  bitset_word *srcp = VBITSET_WORDS (src);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_windex ssize = VBITSET_SIZE (src);
  bitset_windex dsize = VBITSET_SIZE (dst);

  for (unsigned i = 0; i < ssize; i++)
    *dstp++ = ~(*srcp++);

  vbitset_unused_clear (dst);
  memset (dstp + sizeof (bitset_word) * ssize, 0,
          sizeof (bitset_word) * (dsize - ssize));
}


static bool
vbitset_equal_p (bitset dst, bitset src)
{
  bitset_word *srcp = VBITSET_WORDS (src);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_windex ssize = VBITSET_SIZE (src);
  bitset_windex dsize = VBITSET_SIZE (dst);

  unsigned i;
  for (i = 0; i < min (ssize, dsize); i++)
    if (*srcp++ != *dstp++)
      return false;

  if (ssize > dsize)
    {
      for (; i < ssize; i++)
        if (*srcp++)
          return false;
    }
  else
    {
      for (; i < dsize; i++)
        if (*dstp++)
          return false;
    }

  return true;
}


static bool
vbitset_subset_p (bitset dst, bitset src)
{
  bitset_word *srcp = VBITSET_WORDS (src);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_windex ssize = VBITSET_SIZE (src);
  bitset_windex dsize = VBITSET_SIZE (dst);

  unsigned i;
  for (i = 0; i < min (ssize, dsize); i++, dstp++, srcp++)
    if (*dstp != (*srcp | *dstp))
      return false;

  if (ssize > dsize)
    for (; i < ssize; i++)
      if (*srcp++)
        return false;

  return true;
}


static bool
vbitset_disjoint_p (bitset dst, bitset src)
{
  bitset_word *srcp = VBITSET_WORDS (src);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_windex ssize = VBITSET_SIZE (src);
  bitset_windex dsize = VBITSET_SIZE (dst);

  for (unsigned i = 0; i < min (ssize, dsize); i++)
    if (*srcp++ & *dstp++)
      return false;

  return true;
}


static void
vbitset_and (bitset dst, bitset src1, bitset src2)
{
  vbitset_resize (dst, max (BITSET_SIZE_ (src1), BITSET_SIZE_ (src2)));

  bitset_windex dsize = VBITSET_SIZE (dst);
  bitset_windex ssize1 = VBITSET_SIZE (src1);
  bitset_windex ssize2 = VBITSET_SIZE (src2);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);

  for (unsigned i = 0; i < min (ssize1, ssize2); i++)
    *dstp++ = *src1p++ & *src2p++;

  memset (dstp, 0, sizeof (bitset_word) * (dsize - min (ssize1, ssize2)));
}


static bool
vbitset_and_cmp (bitset dst, bitset src1, bitset src2)
{
  vbitset_resize (dst, max (BITSET_SIZE_ (src1), BITSET_SIZE_ (src2)));

  bitset_windex dsize = VBITSET_SIZE (dst);
  bitset_windex ssize1 = VBITSET_SIZE (src1);
  bitset_windex ssize2 = VBITSET_SIZE (src2);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);

  bool changed = false;
  unsigned i;
  for (i = 0; i < min (ssize1, ssize2); i++, dstp++)
    {
      bitset_word tmp = *src1p++ & *src2p++;

      if (*dstp != tmp)
        {
          changed = true;
          *dstp = tmp;
        }
    }

  if (ssize2 > ssize1)
    {
      src1p = src2p;
      ssize1 = ssize2;
    }

  for (; i < ssize1; i++, dstp++)
    {
      if (*dstp != 0)
        {
          changed = true;
          *dstp = 0;
        }
    }

  memset (dstp, 0, sizeof (bitset_word) * (dsize - ssize1));

  return changed;
}


static void
vbitset_andn (bitset dst, bitset src1, bitset src2)
{
  vbitset_resize (dst, max (BITSET_SIZE_ (src1), BITSET_SIZE_ (src2)));

  bitset_windex dsize = VBITSET_SIZE (dst);
  bitset_windex ssize1 = VBITSET_SIZE (src1);
  bitset_windex ssize2 = VBITSET_SIZE (src2);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);

  unsigned i;
  for (i = 0; i < min (ssize1, ssize2); i++)
    *dstp++ = *src1p++ & ~(*src2p++);

  if (ssize2 > ssize1)
    {
      for (; i < ssize2; i++)
        *dstp++ = 0;

      memset (dstp, 0, sizeof (bitset_word) * (dsize - ssize2));
    }
  else
    {
      for (; i < ssize1; i++)
        *dstp++ = *src1p++;

      memset (dstp, 0, sizeof (bitset_word) * (dsize - ssize1));
    }
}


static bool
vbitset_andn_cmp (bitset dst, bitset src1, bitset src2)
{
  vbitset_resize (dst, max (BITSET_SIZE_ (src1), BITSET_SIZE_ (src2)));

  bitset_windex dsize = VBITSET_SIZE (dst);
  bitset_windex ssize1 = VBITSET_SIZE (src1);
  bitset_windex ssize2 = VBITSET_SIZE (src2);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);

  bool changed = false;
  unsigned i;
  for (i = 0; i < min (ssize1, ssize2); i++, dstp++)
    {
      bitset_word tmp = *src1p++ & ~(*src2p++);

      if (*dstp != tmp)
        {
          changed = true;
          *dstp = tmp;
        }
    }

  if (ssize2 > ssize1)
    {
      for (; i < ssize2; i++, dstp++)
        {
          if (*dstp != 0)
            {
              changed = true;
              *dstp = 0;
            }
        }

      memset (dstp, 0, sizeof (bitset_word) * (dsize - ssize2));
    }
  else
    {
      for (; i < ssize1; i++, dstp++)
        {
          bitset_word tmp = *src1p++;

          if (*dstp != tmp)
            {
              changed = true;
              *dstp = tmp;
            }
        }

      memset (dstp, 0, sizeof (bitset_word) * (dsize - ssize1));
    }

  return changed;
}


static void
vbitset_or (bitset dst, bitset src1, bitset src2)
{
  vbitset_resize (dst, max (BITSET_SIZE_ (src1), BITSET_SIZE_ (src2)));

  bitset_windex dsize = VBITSET_SIZE (dst);
  bitset_windex ssize1 = VBITSET_SIZE (src1);
  bitset_windex ssize2 = VBITSET_SIZE (src2);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);

  unsigned i;
  for (i = 0; i < min (ssize1, ssize2); i++)
    *dstp++ = *src1p++ | *src2p++;

  if (ssize2 > ssize1)
    {
      src1p = src2p;
      ssize1 = ssize2;
    }

  for (; i < ssize1; i++)
    *dstp++ = *src1p++;

  memset (dstp, 0, sizeof (bitset_word) * (dsize - ssize1));
}


static bool
vbitset_or_cmp (bitset dst, bitset src1, bitset src2)
{
  vbitset_resize (dst, max (BITSET_SIZE_ (src1), BITSET_SIZE_ (src2)));

  bitset_windex dsize = VBITSET_SIZE (dst);
  bitset_windex ssize1 = VBITSET_SIZE (src1);
  bitset_windex ssize2 = VBITSET_SIZE (src2);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);

  bool changed = false;
  unsigned i;
  for (i = 0; i < min (ssize1, ssize2); i++, dstp++)
    {
      bitset_word tmp = *src1p++ | *src2p++;

      if (*dstp != tmp)
        {
          changed = true;
          *dstp = tmp;
        }
    }

  if (ssize2 > ssize1)
    {
      src1p = src2p;
      ssize1 = ssize2;
    }

  for (; i < ssize1; i++, dstp++)
    {
      bitset_word tmp = *src1p++;

      if (*dstp != tmp)
        {
          changed = true;
          *dstp = tmp;
        }
    }

  memset (dstp, 0, sizeof (bitset_word) * (dsize - ssize1));

  return changed;
}


static void
vbitset_xor (bitset dst, bitset src1, bitset src2)
{
  vbitset_resize (dst, max (BITSET_SIZE_ (src1), BITSET_SIZE_ (src2)));

  bitset_windex dsize = VBITSET_SIZE (dst);
  bitset_windex ssize1 = VBITSET_SIZE (src1);
  bitset_windex ssize2 = VBITSET_SIZE (src2);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);

  unsigned i;
  for (i = 0; i < min (ssize1, ssize2); i++)
    *dstp++ = *src1p++ ^ *src2p++;

  if (ssize2 > ssize1)
    {
      src1p = src2p;
      ssize1 = ssize2;
    }

  for (; i < ssize1; i++)
    *dstp++ = *src1p++;

  memset (dstp, 0, sizeof (bitset_word) * (dsize - ssize1));
}


static bool
vbitset_xor_cmp (bitset dst, bitset src1, bitset src2)
{
  vbitset_resize (dst, max (BITSET_SIZE_ (src1), BITSET_SIZE_ (src2)));

  bitset_windex dsize = VBITSET_SIZE (dst);
  bitset_windex ssize1 = VBITSET_SIZE (src1);
  bitset_windex ssize2 = VBITSET_SIZE (src2);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);

  bool changed = false;
  unsigned i;
  for (i = 0; i < min (ssize1, ssize2); i++, dstp++)
    {
      bitset_word tmp = *src1p++ ^ *src2p++;

      if (*dstp != tmp)
        {
          changed = true;
          *dstp = tmp;
        }
    }

  if (ssize2 > ssize1)
    {
      src1p = src2p;
      ssize1 = ssize2;
    }

  for (; i < ssize1; i++, dstp++)
    {
      bitset_word tmp = *src1p++;

      if (*dstp != tmp)
        {
          changed = true;
          *dstp = tmp;
        }
    }

  memset (dstp, 0, sizeof (bitset_word) * (dsize - ssize1));

  return changed;
}


/* FIXME, these operations need fixing for different size
   bitsets.  */

static void
vbitset_and_or (bitset dst, bitset src1, bitset src2, bitset src3)
{
  if (BITSET_NBITS_ (src1) != BITSET_NBITS_ (src2)
      || BITSET_NBITS_ (src1) != BITSET_NBITS_ (src3))
    {
      bitset_and_or_ (dst, src1, src2, src3);
      return;
    }

  vbitset_resize (dst, BITSET_NBITS_ (src1));

  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);
  bitset_word *src3p = VBITSET_WORDS (src3);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_windex size = VBITSET_SIZE (dst);

  for (unsigned i = 0; i < size; i++)
    *dstp++ = (*src1p++ & *src2p++) | *src3p++;
}


static bool
vbitset_and_or_cmp (bitset dst, bitset src1, bitset src2, bitset src3)
{
  if (BITSET_NBITS_ (src1) != BITSET_NBITS_ (src2)
      || BITSET_NBITS_ (src1) != BITSET_NBITS_ (src3))
    return bitset_and_or_cmp_ (dst, src1, src2, src3);

  vbitset_resize (dst, BITSET_NBITS_ (src1));

  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);
  bitset_word *src3p = VBITSET_WORDS (src3);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_windex size = VBITSET_SIZE (dst);

  bool changed = false;
  for (unsigned i = 0; i < size; i++, dstp++)
    {
      bitset_word tmp = (*src1p++ & *src2p++) | *src3p++;

      if (*dstp != tmp)
        {
          changed = true;
          *dstp = tmp;
        }
    }
  return changed;
}


static void
vbitset_andn_or (bitset dst, bitset src1, bitset src2, bitset src3)
{
  if (BITSET_NBITS_ (src1) != BITSET_NBITS_ (src2)
      || BITSET_NBITS_ (src1) != BITSET_NBITS_ (src3))
    {
      bitset_andn_or_ (dst, src1, src2, src3);
      return;
    }

  vbitset_resize (dst, BITSET_NBITS_ (src1));

  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);
  bitset_word *src3p = VBITSET_WORDS (src3);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_windex size = VBITSET_SIZE (dst);

  for (unsigned i = 0; i < size; i++)
    *dstp++ = (*src1p++ & ~(*src2p++)) | *src3p++;
}


static bool
vbitset_andn_or_cmp (bitset dst, bitset src1, bitset src2, bitset src3)
{
  if (BITSET_NBITS_ (src1) != BITSET_NBITS_ (src2)
      || BITSET_NBITS_ (src1) != BITSET_NBITS_ (src3))
    return bitset_andn_or_cmp_ (dst, src1, src2, src3);

  vbitset_resize (dst, BITSET_NBITS_ (src1));

  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);
  bitset_word *src3p = VBITSET_WORDS (src3);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_windex size = VBITSET_SIZE (dst);

  bool changed = false;
  for (unsigned i = 0; i < size; i++, dstp++)
    {
      bitset_word tmp = (*src1p++ & ~(*src2p++)) | *src3p++;

      if (*dstp != tmp)
        {
          changed = true;
          *dstp = tmp;
        }
    }
  return changed;
}


static void
vbitset_or_and (bitset dst, bitset src1, bitset src2, bitset src3)
{
  if (BITSET_NBITS_ (src1) != BITSET_NBITS_ (src2)
      || BITSET_NBITS_ (src1) != BITSET_NBITS_ (src3))
    {
      bitset_or_and_ (dst, src1, src2, src3);
      return;
    }

  vbitset_resize (dst, BITSET_NBITS_ (src1));

  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);
  bitset_word *src3p = VBITSET_WORDS (src3);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_windex size = VBITSET_SIZE (dst);

  for (unsigned i = 0; i < size; i++)
    *dstp++ = (*src1p++ | *src2p++) & *src3p++;
}


static bool
vbitset_or_and_cmp (bitset dst, bitset src1, bitset src2, bitset src3)
{
  if (BITSET_NBITS_ (src1) != BITSET_NBITS_ (src2)
      || BITSET_NBITS_ (src1) != BITSET_NBITS_ (src3))
    return bitset_or_and_cmp_ (dst, src1, src2, src3);

  vbitset_resize (dst, BITSET_NBITS_ (src1));

  bitset_word *src1p = VBITSET_WORDS (src1);
  bitset_word *src2p = VBITSET_WORDS (src2);
  bitset_word *src3p = VBITSET_WORDS (src3);
  bitset_word *dstp = VBITSET_WORDS (dst);
  bitset_windex size = VBITSET_SIZE (dst);

  bool changed = false;
  unsigned i;
  for (i = 0; i < size; i++, dstp++)
    {
      bitset_word tmp = (*src1p++ | *src2p++) & *src3p++;

      if (*dstp != tmp)
        {
          changed = true;
          *dstp = tmp;
        }
    }
  return changed;
}


static void
vbitset_copy (bitset dst, bitset src)
{
  if (BITSET_COMPATIBLE_ (dst, src))
    vbitset_copy1 (dst, src);
  else
    bitset_copy_ (dst, src);
}


static void
vbitset_free (bitset bset)
{
  free (VBITSET_WORDS (bset));
}


/* Vector of operations for multiple word bitsets.  */
struct bitset_vtable vbitset_vtable = {
  vbitset_set,
  vbitset_reset,
  bitset_toggle_,
  vbitset_test,
  vbitset_resize,
  bitset_size_,
  bitset_count_,
  vbitset_empty_p,
  vbitset_ones,
  vbitset_zero,
  vbitset_copy,
  vbitset_disjoint_p,
  vbitset_equal_p,
  vbitset_not,
  vbitset_subset_p,
  vbitset_and,
  vbitset_and_cmp,
  vbitset_andn,
  vbitset_andn_cmp,
  vbitset_or,
  vbitset_or_cmp,
  vbitset_xor,
  vbitset_xor_cmp,
  vbitset_and_or,
  vbitset_and_or_cmp,
  vbitset_andn_or,
  vbitset_andn_or_cmp,
  vbitset_or_and,
  vbitset_or_and_cmp,
  vbitset_list,
  vbitset_list_reverse,
  vbitset_free,
  BITSET_VECTOR
};


size_t
vbitset_bytes (bitset_bindex n_bits ATTRIBUTE_UNUSED)
{
  return sizeof (struct vbitset_struct);
}


bitset
vbitset_init (bitset bset, bitset_bindex n_bits)
{
  bset->b.vtable = &vbitset_vtable;
  bset->b.cindex = 0;
  VBITSET_SIZE (bset) = 0;
  vbitset_resize (bset, n_bits);
  return bset;
}
